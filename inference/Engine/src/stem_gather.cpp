#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>

#include <cstdint>
#include <cstring>

std::tuple<int64_t, int64_t> stem_gather_cpu_fast(
    torch::Tensor token_ids,
    torch::Tensor cpu_tbl,     // [vocab, L, D] bf16 contiguous
    torch::Tensor stage,       // [L, Umax, D] bf16 contiguous (pinned ok)
    torch::Tensor inv,         // [Tmax] int32 pinned
    torch::Tensor unique_out,  // [Umax] int32
    torch::Tensor seen,        // [vocab] int32
    torch::Tensor slot,        // [vocab] int32
    int32_t epoch
) {
  TORCH_CHECK(!token_ids.is_cuda(), "token_ids must be CPU");
  TORCH_CHECK(!cpu_tbl.is_cuda(), "cpu_tbl must be CPU");
  TORCH_CHECK(!stage.is_cuda(), "stage must be CPU");
  TORCH_CHECK(!inv.is_cuda(), "inv must be CPU");

  TORCH_CHECK(cpu_tbl.scalar_type() == at::kBFloat16, "cpu_tbl must be bfloat16");
  TORCH_CHECK(stage.scalar_type() == at::kBFloat16, "stage must be bfloat16");
  TORCH_CHECK(inv.scalar_type() == at::kInt, "inv must be int32");
  TORCH_CHECK(unique_out.scalar_type() == at::kInt, "unique_out must be int32");
  TORCH_CHECK(seen.scalar_type() == at::kInt, "seen must be int32");
  TORCH_CHECK(slot.scalar_type() == at::kInt, "slot must be int32");

  TORCH_CHECK(cpu_tbl.dim() == 3, "cpu_tbl must be [vocab, L, D]");
  TORCH_CHECK(stage.dim() == 3, "stage must be [L, Umax, D]");
  TORCH_CHECK(inv.dim() == 1, "inv must be 1D");

  // Ensure contiguous to enable memcpy fast path
  cpu_tbl = cpu_tbl.contiguous();
  stage = stage.contiguous();

  token_ids = token_ids.contiguous().view({-1});
  const int64_t T = token_ids.numel();
  TORCH_CHECK(T <= inv.numel(), "inv buffer too small");

  const int64_t vocab = cpu_tbl.size(0);
  const int64_t L = cpu_tbl.size(1);
  const int64_t D = cpu_tbl.size(2);
  TORCH_CHECK(stage.size(0) == L && stage.size(2) == D, "stage shape mismatch");
  const int64_t Umax = stage.size(1);

  auto inv_ptr = inv.data_ptr<int32_t>();
  auto seen_ptr = seen.data_ptr<int32_t>();
  auto slot_ptr = slot.data_ptr<int32_t>();
  auto uniq_ptr = unique_out.data_ptr<int32_t>();

  int64_t U = 0;

  // ---- Pass 1: unique + inverse (single-thread, fast O(T)) ----
  if (token_ids.scalar_type() == at::kLong) {
    auto ids = token_ids.data_ptr<int64_t>();
    for (int64_t t = 0; t < T; ++t) {
      int64_t id64 = ids[t];
      TORCH_CHECK(id64 >= 0 && id64 < vocab, "token id out of range");
      int32_t id = static_cast<int32_t>(id64);
      if (seen_ptr[id] != epoch) {
        TORCH_CHECK(U < Umax, "Too many unique tokens for stage");
        seen_ptr[id] = epoch;
        slot_ptr[id] = static_cast<int32_t>(U);
        uniq_ptr[U] = id;
        ++U;
      }
      inv_ptr[t] = slot_ptr[id];
    }
  } else if (token_ids.scalar_type() == at::kInt) {
    auto ids = token_ids.data_ptr<int32_t>();
    for (int64_t t = 0; t < T; ++t) {
      int32_t id = ids[t];
      TORCH_CHECK(id >= 0 && id < vocab, "token id out of range");
      if (seen_ptr[id] != epoch) {
        TORCH_CHECK(U < Umax, "Too many unique tokens for stage");
        seen_ptr[id] = epoch;
        slot_ptr[id] = static_cast<int32_t>(U);
        uniq_ptr[U] = id;
        ++U;
      }
      inv_ptr[t] = slot_ptr[id];
    }
  } else {
    TORCH_CHECK(false, "token_ids must be int32 or int64");
  }

  // ---- Pass 2: memcpy copy rows ----
  // cpu_tbl layout contiguous: [tok][l][d]
  // stage layout contiguous:   [l][u][d]
  const c10::BFloat16* tbl = (const c10::BFloat16*)cpu_tbl.data_ptr<at::BFloat16>();
  c10::BFloat16* st = (c10::BFloat16*)stage.data_ptr<at::BFloat16>();

  // Strides for contiguous tensors:
  // tbl offset = tok*(L*D) + l*D
  // st  offset = l*(Umax*D) + u*D
  const int64_t tbl_tok_stride = L * D;
  const int64_t tbl_l_stride = D;
  const int64_t st_l_stride = Umax * D;
  const int64_t st_u_stride = D;

  const size_t row_bytes = (size_t)D * sizeof(c10::BFloat16);

  at::parallel_for(0, U, 1, [&](int64_t u0, int64_t u1) {
    for (int64_t u = u0; u < u1; ++u) {
      const int32_t tok = uniq_ptr[u];
      const c10::BFloat16* src_tok = tbl + (int64_t)tok * tbl_tok_stride;
      for (int64_t l = 0; l < L; ++l) {
        const c10::BFloat16* src = src_tok + l * tbl_l_stride;
        c10::BFloat16* dst = st + l * st_l_stride + u * st_u_stride;
        std::memcpy(dst, src, row_bytes);
      }
    }
  });

  return {U, T};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stem_gather_cpu", &stem_gather_cpu_fast,
        "STEM fused dedup + gather into pinned stage (CPU, memcpy fast)");
}
