#pragma once

#include <cassert>
#include <cstdint>
#include <list>
#include <unordered_map>

// Simple LFU cache for mapping token_id -> fixed GPU slot in [0, capacity).
// API
//   - lookup_slot(key, slot_out): returns true if key is in cache, sets
//   slot_out.
//   - touch(key): ensure key is in cache, updating its frequency and possibly
//                 evicting another key if the cache is full.
// Eviction policy:
//   - Always evict the key with the lowest frequency (min_freq_).
//   - Within the same frequency bucket, evict the least-recently-used key
//     (tail of the list), so it's LFU with LRU tie-breaking.
class LFUCache {
    public:
      explicit LFUCache(int capacity)
          : capacity_(capacity), size_(0), next_free_slot_(0), min_freq_(0) {}
    
      // If already present:
      //   - Increment its frequency and update the frequency buckets.
      //   - Return 1 (hit).
      // If not present:
      //   - If cache not full, allocate a new slot [0, capacity_).
      //   - If cache is full, evict the LFU key (with LRU tie-break),
      //     reusing its slot for 'key'.
      //   - Insert 'key' into the cache, with frequency = 1.
      //   - Return 0 (miss).
      // After this call (when capacity_ > 0), key is guaranteed to have a slot.
      uint8_t touch(int64_t key, int &slot_out) {
        auto it = entries_.find(key);
        if (it != entries_.end()) {
          // Existing key: bump its frequency.
          increment_freq(it->first, it->second);
          slot_out = it->second.slot;
          return 1;
        }
    
        // New key insert.
        if (size_ < capacity_) {
          // Still have free slots: assign next_free_slot_.
          int slot = next_free_slot_++;
          insert_new(key, slot);
          slot_out = slot;
        } else {
          // Cache full: evict LFU entry, reuse its slot.
          int victim_slot = -1;
          int64_t victim_key = 0;
          evict_one(victim_key, victim_slot);
          insert_new(key, victim_slot);
          slot_out = victim_slot;
        }
        return 0;
      }
    
      int capacity() const { return capacity_; }
      int size() const { return size_; }
    
    private:
      struct Entry {
        int slot;                        // GPU cache slot index [0, capacity_)
        int freq;                        // frequency count
        std::list<int64_t>::iterator it; // iterator into freq_buckets_[freq]
      };
    
      int capacity_;
      int size_;
      int next_free_slot_;
      int min_freq_;
    
      // key -> {slot, freq, iterator into freq_buckets_[freq]}
      std::unordered_map<int64_t, Entry> entries_;
    
      // freq -> list of keys with that frequency.
      // We use a list so we can do O(1) LRU eviction within each freq bucket.
      std::unordered_map<int, std::list<int64_t>> freq_buckets_;
    
      // Insert a brand-new key with frequency = 1, assigned to 'slot'.
      void insert_new(int64_t key, int slot) {
        min_freq_ = 1; // new entry with freq=1 is now the minimum
        auto &lst = freq_buckets_[1];
        lst.push_front(key);
    
        Entry e;
        e.slot = slot;
        e.freq = 1;
        e.it = lst.begin();
    
        entries_[key] = std::move(e);
    
        if (size_ < capacity_) {
          ++size_;
        }
      }
    
      // Evict a single key from the cache (the LFU one).
      // Returns the evicted key and its slot via references.
      void evict_one(int64_t &victim_key, int &victim_slot) {
        // Find the bucket with min_freq_
        auto fb_it = freq_buckets_.find(min_freq_);
        assert(fb_it != freq_buckets_.end());
        auto &lst = fb_it->second;
        assert(!lst.empty());
    
        // Evict from the back to get LRU among this frequency.
        victim_key = lst.back();
        lst.pop_back();
    
        if (lst.empty()) {
          freq_buckets_.erase(fb_it);
          // min_freq_ will be reset to 1 in insert_new().
          // (We don't need to scan frequencies here.)
        }
    
        auto e_it = entries_.find(victim_key);
        assert(e_it != entries_.end());
        victim_slot = e_it->second.slot;
    
        entries_.erase(e_it);
        // size_ stays the same: we immediately insert a new key in this slot.
      }
    
      // Internal helper: increment frequency of an existing entry.
      void increment_freq(int64_t key, Entry &entry) {
        int old_freq = entry.freq;
        int new_freq = old_freq + 1;
    
        // Remove from old frequency list.
        auto fb_it = freq_buckets_.find(old_freq);
        assert(fb_it != freq_buckets_.end());
        auto &old_list = fb_it->second;
        old_list.erase(entry.it);
        if (old_list.empty()) {
          freq_buckets_.erase(fb_it);
          if (min_freq_ == old_freq) {
            // This frequency bucket is now empty; min_freq_ increases.
            ++min_freq_;
          }
        }
    
        // Add to new frequency list.
        auto &new_list = freq_buckets_[new_freq];
        new_list.push_front(key);
        entry.freq = new_freq;
        entry.it = new_list.begin();
      }
    };