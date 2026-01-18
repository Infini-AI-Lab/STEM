#include "lfu_cache.h"
#include <torch/extension.h>

#include <cstdint>
#include <unordered_map>

namespace {

struct STEMLFUCache {
  STEMLFUCache(int capacity, int64_t hidden_dim)
      : capacity_(capacity), hidden_dim_(hidden_dim), lfu_(capacity) {
    TORCH_CHECK(capacity_ > 0, "STEMLFUCache capacity must be > 0");

    // slot -> token_id mapping (for sanity/debug), kept on CPU
    token_ids_ = torch::full(
        {capacity_}, static_cast<int64_t>(-1),
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  }

  // ------------------------------------------------------------------------
  // Public interface
  // ------------------------------------------------------------------------

  // Returns:
  //   buffer_ids, src_offsets, dst_offsets, miss_cnt
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t>
  plan_offsets(torch::Tensor token_ids) {
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be 1D");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt64,
                "token_ids must be int64");

    auto token_ids_cpu = token_ids.to(torch::kCPU).contiguous();
    const auto B = token_ids_cpu.size(0);
    const auto *ids_ptr = token_ids_cpu.data_ptr<int64_t>();

    // ------------------------
    // Build outputs: buffer_ids / offsets / hit_cnt
    // buffer_ids: indices where the STEMs will reside in the GPU buffer (after
    // LFU caching) offsets: indices to read from the CPU STEM table
    // ------------------------
    torch::Tensor buffer_ids = torch::empty(
        {B}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    torch::Tensor src_offsets = torch::empty(
        {B}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    torch::Tensor dst_offsets = torch::empty(
        {B}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    auto *buffer_ids_ptr = buffer_ids.data_ptr<int32_t>();
    auto *src_offsets_ptr = src_offsets.data_ptr<int32_t>();
    auto *dst_offsets_ptr = dst_offsets.data_ptr<int32_t>();

    int64_t miss_cnt = 0;

    for (int64_t i = 0; i < B; ++i) {
      int64_t tok = ids_ptr[i];

      int new_slot = -1;
      bool was_present = lfu_.touch(tok, new_slot);
      if (!was_present) {
        src_offsets_ptr[miss_cnt] = tok;
        dst_offsets_ptr[miss_cnt++] = new_slot;
        // Update slot -> token_id mapping for debugging
        token_ids_.index_put_({new_slot}, tok);
      }
      buffer_ids_ptr[i] = new_slot;
    }

    // Slice offset tensors to only include valid entries
    return std::make_tuple(buffer_ids, src_offsets.slice(0, 0, miss_cnt),
                           dst_offsets.slice(0, 0, miss_cnt), miss_cnt);
  }

  // Optional: slot->token mapping (CPU)
  torch::Tensor slot_token_ids() const { return token_ids_; }

private:
  int capacity_;
  int64_t hidden_dim_;
  LFUCache lfu_;

  torch::Tensor token_ids_; // [capacity], int64, on CPU (slot -> token_id)
};

} // namespace

// -------------------- PYBIND11 BINDINGS --------------------

void register_stem_lfu_cache_bindings(py::module &m) {
  py::class_<STEMLFUCache>(m, "STEMLFUCache")
      .def(py::init<int, int64_t>(), py::arg("capacity"), py::arg("hidden_dim"))
      .def("plan_offsets", &STEMLFUCache::plan_offsets, py::arg("token_ids"),
           R"doc(
Plan offsets for a batch of token IDs (compat version).

Returns:
  buffer_ids:  (B,) int32 tensor (CPU). Token positions in the GPU STEM buffer.
  src_offsets: (miss_cnt,) int32 tensor (CPU). Tokens to prefetch from CPU STEM table.
  dst_offsets: (miss_cnt,) int32 tensor (CPU). Destination slots in the GPU STEM buffer.
  miss_cnt:    Python int. Number of GPU buffer misses (items to prefetch).
)doc")
      .def_property_readonly("slot_token_ids", &STEMLFUCache::slot_token_ids,
                             R"doc(
CPU tensor [capacity] mapping GPU slot -> token_id (for debugging).
)doc");
}