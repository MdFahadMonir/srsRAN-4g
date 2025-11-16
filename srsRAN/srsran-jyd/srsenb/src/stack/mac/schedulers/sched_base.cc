/**
 * Copyright 2013-2021 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include "srsenb/hdr/stack/mac/schedulers/sched_base.h"
#include "srsenb/hdr/stack/mac/sched.h"
#include "srsenb/hdr/stack/mac/sched_interface.h"

namespace srsenb {

int get_ue_cc_idx_if_pdsch_enabled(const sched_ue& user, sf_sched* tti_sched)
{
  // Do not allocate a user multiple times in the same tti
  if (tti_sched->is_dl_alloc(user.get_rnti())) {
    return -1;
  }
  // Do not allocate a user to an inactive carrier
  auto p = user.get_active_cell_index(tti_sched->get_enb_cc_idx());
  if (not p.first) {
    return -1;
  }
  uint32_t cell_idx = p.second;
  // Do not allow allocations when PDSCH is deactivated
  if (not user.pdsch_enabled(tti_sched->get_tti_rx(), tti_sched->get_enb_cc_idx())) {
    return -1;
  }
  return cell_idx;
}
const dl_harq_proc* get_dl_retx_harq(sched_ue& user, sf_sched* tti_sched)
{
  if (get_ue_cc_idx_if_pdsch_enabled(user, tti_sched) < 0) {
    return nullptr;
  }
  dl_harq_proc* h = user.get_pending_dl_harq(tti_sched->get_tti_tx_dl(), tti_sched->get_enb_cc_idx());
  return h;
}
const dl_harq_proc* get_dl_newtx_harq(sched_ue& user, sf_sched* tti_sched)
{
  if (get_ue_cc_idx_if_pdsch_enabled(user, tti_sched) < 0) {
    return nullptr;
  }
  return user.get_empty_dl_harq(tti_sched->get_tti_tx_dl(), tti_sched->get_enb_cc_idx());
}

alloc_result try_dl_retx_alloc(sf_sched& tti_sched, sched_ue& ue, const dl_harq_proc& h)
{
  // Try to reuse the same mask
  rbgmask_t    retx_mask = h.get_rbgmask();
  alloc_result code      = tti_sched.alloc_dl_user(&ue, retx_mask, h.get_id());
  if (code != alloc_result::sch_collision) {
    return code;
  }

  // If previous mask does not fit, find another with exact same number of rbgs
  size_t nof_rbg             = retx_mask.count();
  bool   is_contiguous_alloc = ue.get_dci_format() == SRSRAN_DCI_FORMAT1A;
  retx_mask                  = find_available_rbgmask(nof_rbg, is_contiguous_alloc, tti_sched.get_dl_mask());
  if (retx_mask.count() == nof_rbg) {
    return tti_sched.alloc_dl_user(&ue, retx_mask, h.get_id());
  }
  return alloc_result::sch_collision;
}

int cmpfunc(const void* a, const void* b)
{
  const fr_slice_t* slice_x = (fr_slice_t*) a;
  const fr_slice_t* slice_y = (fr_slice_t*) b;
  const int* x = (int*) &slice_x->id;
  const int* y = (int*) &slice_y->id;
  return (*x - *y);
}

alloc_result try_dl_newtx_alloc_greedy(sf_sched& tti_sched, sched_ue& ue, const dl_harq_proc& h, rbgmask_t* result_mask)
{
  if (result_mask != nullptr) {
    *result_mask = {};
  }

  // Get slice stats
  Slicing &slice_stats = Slicing::getInstance();
  ul_dl_slice_conf_t* stats_dl = &slice_stats.stats_slice_conf.dl;
  ue_slice_conf_t* stats_ue_s = &slice_stats.stats_ue_slice_conf;

  // If no slice in enb, schedule UE by orginal RBG mask
  rbgmask_t slice_mask = tti_sched.get_dl_mask();

  // If slice is enabled in enb, check UE associated slice id exists
  int idx = ue.slice_id();
  #ifdef ENABLE_SLICER
    uint64_t ue_imsi = imsiTracker.find_imsi(ue.get_rnti());
    if(ue_imsi > 0)
    {
      idx = (int) (ue_imsi % 3);
      //int idx2 = (int) (ue_imsi % 3);
      // printf("[slicer] UE: %015" PRIu64 " associated to slice id: %d\n", ue_imsi, idx);
    }
    else
      idx = -1;
  #endif
  // int idx = 2;
  if (stats_dl->len_slices > 0 && idx >= 0) {
    // Get currently slice pointer by binary search
    fr_slice_t* pslice = (fr_slice_t*) std::bsearch(&idx, stats_dl->slices, stats_dl->len_slices, sizeof(stats_dl->slices[0]), cmpfunc);
    // if slice exists, get algo data
    if (pslice) {
      static_slice_t* sta = &pslice->params.u.sta;
      // if current ue associated to slice, schedule UE by the RBG mask with static slice algo
      if (idx != -1) {
        size_t l1 = sta->pos_low;
        size_t l2 = sta->pos_high;
        // mark already used RB
        slice_mask.fill(0, l1, true);
        slice_mask.fill(l2, 13, true); // TODO: Get DL PRB (n_prb) from config
      }
    }
  }

  // If all RBGs are occupied, the next steps can be shortcut
  if (slice_mask.all()) {
    return alloc_result::no_sch_space;
  }

  const rbgmask_t& current_mask = slice_mask;

  // If there is no data to transmit, no need to allocate
  srsran::interval<uint32_t> req_bytes = ue.get_requested_dl_bytes(tti_sched.get_enb_cc_idx());
  if (req_bytes.stop() == 0) {
    return alloc_result::no_rnti_opportunity;
  }

  sched_ue_cell* ue_cell = ue.find_ue_carrier(tti_sched.get_enb_cc_idx());
  srsran_assert(ue_cell != nullptr, "dl newtx alloc called for invalid cell");
  srsran_dci_format_t dci_format = ue.get_dci_format();
  tbs_info            tb;
  rbgmask_t           opt_mask;
  if (not find_optimal_rbgmask(
          *ue_cell, tti_sched.get_tti_tx_dl(), current_mask, dci_format, req_bytes, tb, opt_mask)) {
    return alloc_result::no_sch_space;
  }

  // empty RBGs were found. Attempt allocation
  alloc_result ret = tti_sched.alloc_dl_user(&ue, opt_mask, h.get_id());
  if (ret == alloc_result::success and result_mask != nullptr) {
    *result_mask = opt_mask;
  }
  return ret;
}

/*****************
 *  UL Helpers
 ****************/

int get_ue_cc_idx_if_pusch_enabled(const sched_ue& user, sf_sched* tti_sched, bool needs_pdcch)
{
  // Do not allocate a user multiple times in the same tti
  if (tti_sched->is_ul_alloc(user.get_rnti())) {
    return -1;
  }
  // Do not allocate a user to an inactive carrier
  auto p = user.get_active_cell_index(tti_sched->get_enb_cc_idx());
  if (not p.first) {
    return -1;
  }
  uint32_t cell_idx = p.second;
  // Do not allow allocations when PDSCH is deactivated
  if (not user.pusch_enabled(tti_sched->get_tti_rx(), tti_sched->get_enb_cc_idx(), needs_pdcch)) {
    return -1;
  }
  return cell_idx;
}
const ul_harq_proc* get_ul_retx_harq(sched_ue& user, sf_sched* tti_sched)
{
  if (get_ue_cc_idx_if_pusch_enabled(user, tti_sched, false) < 0) {
    return nullptr;
  }
  const ul_harq_proc* h = user.get_ul_harq(tti_sched->get_tti_tx_ul(), tti_sched->get_enb_cc_idx());
  return h->has_pending_retx() ? h : nullptr;
}
const ul_harq_proc* get_ul_newtx_harq(sched_ue& user, sf_sched* tti_sched)
{
  if (get_ue_cc_idx_if_pusch_enabled(user, tti_sched, true) < 0) {
    return nullptr;
  }
  const ul_harq_proc* h = user.get_ul_harq(tti_sched->get_tti_tx_ul(), tti_sched->get_enb_cc_idx());
  return h->is_empty() ? h : nullptr;
}

alloc_result try_ul_retx_alloc(sf_sched& tti_sched, sched_ue& ue, const ul_harq_proc& h)
{
  prb_interval alloc = h.get_alloc();
  if (tti_sched.get_cc_cfg()->nof_prb() == 6 and h.is_msg3()) {
    // We allow collisions with PUCCH for special case of Msg3 and 6 PRBs
    return tti_sched.alloc_ul_user(&ue, alloc);
  }

  // If can schedule the same mask as in earlier tx, do it
  if (not tti_sched.get_ul_mask().any(alloc.start(), alloc.stop())) {
    alloc_result ret = tti_sched.alloc_ul_user(&ue, alloc);
    if (ret != alloc_result::sch_collision) {
      return ret;
    }
  }

  // Avoid measGaps accounting for PDCCH
  if (not ue.pusch_enabled(tti_sched.get_tti_rx(), tti_sched.get_enb_cc_idx(), true)) {
    return alloc_result::no_rnti_opportunity;
  }
  uint32_t nof_prbs = alloc.length();
  alloc             = find_contiguous_ul_prbs(nof_prbs, tti_sched.get_ul_mask());
  if (alloc.length() != nof_prbs) {
    return alloc_result::no_sch_space;
  }
  return tti_sched.alloc_ul_user(&ue, alloc);
}

} // namespace srsenb
