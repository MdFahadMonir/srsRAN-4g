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

#include <srsenb/hdr/stack/mac/sched_ue.h>
#include <string.h>

#include "srsenb/hdr/stack/mac/sched.h"
#include "srsenb/hdr/stack/mac/sched_carrier.h"
#include "srsenb/hdr/stack/mac/sched_helpers.h"
#include "srsran/srslog/srslog.h"
#include "srsran/support/srsran_assert.h"
#include "srsran/common/standard_streams.h"

#define Console(fmt, ...) srsran::console(fmt, ##__VA_ARGS__)
#define Error(fmt, ...) srslog::fetch_basic_logger("MAC").error(fmt, ##__VA_ARGS__)

using srsran::tti_point;

namespace srsenb {

/*******************************************************
 *
 * Initialization and sched configuration functions
 *
 *******************************************************/

sched::sched() {}

sched::~sched() {}

void sched::init(rrc_interface_mac* rrc_, const sched_args_t& sched_cfg_)
{
  rrc       = rrc_;
  sched_cfg = sched_cfg_;

  // Initialize first carrier scheduler
  carrier_schedulers.emplace_back(new carrier_sched{rrc, &ue_db, 0, &sched_results});

  reset();
}

int sched::reset()
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  for (std::unique_ptr<carrier_sched>& c : carrier_schedulers) {
    c->reset();
  }
  ue_db.clear();
  return 0;
}

/// Called by rrc::init
int sched::cell_cfg(const std::vector<sched_interface::cell_cfg_t>& cell_cfg)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  // Setup derived config params
  sched_cell_params.resize(cell_cfg.size());
  for (uint32_t cc_idx = 0; cc_idx < cell_cfg.size(); ++cc_idx) {
    if (not sched_cell_params[cc_idx].set_cfg(cc_idx, cell_cfg[cc_idx], sched_cfg)) {
      return SRSRAN_ERROR;
    }
  }

  sched_results.set_nof_carriers(cell_cfg.size());

  // Create remaining cells, if not created yet
  uint32_t prev_size = carrier_schedulers.size();
  carrier_schedulers.resize(sched_cell_params.size());
  for (uint32_t i = prev_size; i < sched_cell_params.size(); ++i) {
    carrier_schedulers[i].reset(new carrier_sched{rrc, &ue_db, i, &sched_results});
  }

  // setup all carriers cfg params
  for (uint32_t i = 0; i < sched_cell_params.size(); ++i) {
    carrier_schedulers[i]->carrier_cfg(sched_cell_params[i]);
  }

  configured = true;
  return 0;
}

/*******************************************************
 *
 * FAPI-like main sched interface. Wrappers to UE object
 *
 *******************************************************/

int sched::ue_cfg(uint16_t rnti, const sched_interface::ue_cfg_t& ue_cfg)
{
  {
    // config existing user
    std::lock_guard<std::mutex> lock(sched_mutex);
    auto                        it = ue_db.find(rnti);
    if (it != ue_db.end()) {
      it->second->set_cfg(ue_cfg);
      return SRSRAN_SUCCESS;
    }
  }

  // Add new user case
  std::unique_ptr<sched_ue>   ue{new sched_ue(rnti, sched_cell_params, ue_cfg)};
  std::lock_guard<std::mutex> lock(sched_mutex);
  ue_db.insert(rnti, std::move(ue));
  return SRSRAN_SUCCESS;
}

int sched::ue_rem(uint16_t rnti)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  if (ue_db.contains(rnti)) {
    ue_db.erase(rnti);
    // TODO: remove ue from ue slice stats
  } else {
    Error("User rnti=0x%x not found", rnti);
    return SRSRAN_ERROR;
  }
  return SRSRAN_SUCCESS;
}

bool sched::ue_exists(uint16_t rnti)
{
  return ue_db_access_locked(
             rnti, [](sched_ue& ue) {}, nullptr, false) >= 0;
}

void sched::phy_config_enabled(uint16_t rnti, bool enabled)
{
  // TODO: Check if correct use of last_tti
  ue_db_access_locked(
      rnti, [this, enabled](sched_ue& ue) { ue.phy_config_enabled(last_tti, enabled); }, __PRETTY_FUNCTION__);
}

int sched::bearer_ue_cfg(uint16_t rnti, uint32_t lc_id, const mac_lc_ch_cfg_t& cfg_)
{
  return ue_db_access_locked(rnti, [lc_id, cfg_](sched_ue& ue) { ue.set_bearer_cfg(lc_id, cfg_); });
}

int sched::bearer_ue_rem(uint16_t rnti, uint32_t lc_id)
{
  return ue_db_access_locked(rnti, [lc_id](sched_ue& ue) { ue.rem_bearer(lc_id); });
}

uint32_t sched::get_dl_buffer(uint16_t rnti)
{
  uint32_t ret = SRSRAN_ERROR;
  ue_db_access_locked(
      rnti, [&ret](sched_ue& ue) { ret = ue.get_pending_dl_rlc_data(); }, __PRETTY_FUNCTION__);
  return ret;
}

uint32_t sched::get_ul_buffer(uint16_t rnti)
{
  // TODO: Check if correct use of last_tti
  uint32_t ret = SRSRAN_ERROR;
  ue_db_access_locked(
      rnti,
      [this, &ret](sched_ue& ue) { ret = ue.get_pending_ul_new_data(to_tx_ul(last_tti), -1); },
      __PRETTY_FUNCTION__);
  return ret;
}

int sched::dl_rlc_buffer_state(uint16_t rnti, uint32_t lc_id, uint32_t tx_queue, uint32_t prio_tx_queue)
{
  return ue_db_access_locked(rnti, [&](sched_ue& ue) { ue.dl_buffer_state(lc_id, tx_queue, prio_tx_queue); });
}

int sched::dl_mac_buffer_state(uint16_t rnti, uint32_t ce_code, uint32_t nof_cmds)
{
  return ue_db_access_locked(rnti, [ce_code, nof_cmds](sched_ue& ue) { ue.mac_buffer_state(ce_code, nof_cmds); });
}

int sched::dl_ack_info(uint32_t tti_rx, uint16_t rnti, uint32_t enb_cc_idx, uint32_t tb_idx, bool ack)
{
  int ret = -1;
  ue_db_access_locked(
      rnti,
      [&](sched_ue& ue) { ret = ue.set_ack_info(tti_point{tti_rx}, enb_cc_idx, tb_idx, ack); },
      __PRETTY_FUNCTION__);
  return ret;
}

int sched::ul_crc_info(uint32_t tti_rx, uint16_t rnti, uint32_t enb_cc_idx, bool crc)
{
  return ue_db_access_locked(
      rnti, [tti_rx, enb_cc_idx, crc](sched_ue& ue) { ue.set_ul_crc(tti_point{tti_rx}, enb_cc_idx, crc); });
}

int sched::dl_ri_info(uint32_t tti, uint16_t rnti, uint32_t enb_cc_idx, uint32_t ri_value)
{
  return ue_db_access_locked(
      rnti, [tti, enb_cc_idx, ri_value](sched_ue& ue) { ue.set_dl_ri(tti_point{tti}, enb_cc_idx, ri_value); });
}

int sched::dl_pmi_info(uint32_t tti, uint16_t rnti, uint32_t enb_cc_idx, uint32_t pmi_value)
{
  return ue_db_access_locked(
      rnti, [tti, enb_cc_idx, pmi_value](sched_ue& ue) { ue.set_dl_pmi(tti_point{tti}, enb_cc_idx, pmi_value); });
}

int sched::dl_cqi_info(uint32_t tti, uint16_t rnti, uint32_t enb_cc_idx, uint32_t cqi_value)
{
  return ue_db_access_locked(
      rnti, [tti, enb_cc_idx, cqi_value](sched_ue& ue) { ue.set_dl_cqi(tti_point{tti}, enb_cc_idx, cqi_value); });
}

int sched::dl_sb_cqi_info(uint32_t tti, uint16_t rnti, uint32_t enb_cc_idx, uint32_t sb_idx, uint32_t cqi_value)
{
  return ue_db_access_locked(rnti, [tti, enb_cc_idx, cqi_value, sb_idx](sched_ue& ue) {
    ue.set_dl_sb_cqi(tti_point{tti}, enb_cc_idx, sb_idx, cqi_value);
  });
}

int sched::dl_rach_info(uint32_t enb_cc_idx, dl_sched_rar_info_t rar_info)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  return carrier_schedulers[enb_cc_idx]->dl_rach_info(rar_info);
}

int sched::ul_snr_info(uint32_t tti_rx, uint16_t rnti, uint32_t enb_cc_idx, float snr, uint32_t ul_ch_code)
{
  return ue_db_access_locked(rnti,
                             [&](sched_ue& ue) { ue.set_ul_snr(tti_point{tti_rx}, enb_cc_idx, snr, ul_ch_code); });
}

int sched::ul_bsr(uint16_t rnti, uint32_t lcg_id, uint32_t bsr)
{
  return ue_db_access_locked(rnti, [lcg_id, bsr](sched_ue& ue) { ue.ul_buffer_state(lcg_id, bsr); });
}

int sched::ul_buffer_add(uint16_t rnti, uint32_t lcid, uint32_t bytes)
{
  return ue_db_access_locked(rnti, [lcid, bytes](sched_ue& ue) { ue.ul_buffer_add(lcid, bytes); });
}

int sched::ul_phr(uint16_t rnti, int phr, uint32_t ul_nof_prb)
{
  return ue_db_access_locked(
      rnti, [phr, ul_nof_prb](sched_ue& ue) { ue.ul_phr(phr, ul_nof_prb); }, __PRETTY_FUNCTION__);
}

int sched::ul_sr_info(uint32_t tti, uint16_t rnti)
{
  return ue_db_access_locked(
      rnti, [](sched_ue& ue) { ue.set_sr(); }, __PRETTY_FUNCTION__);
}

void sched::set_dl_tti_mask(uint8_t* tti_mask, uint32_t nof_sfs)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  carrier_schedulers[0]->set_dl_tti_mask(tti_mask, nof_sfs);
}

std::array<int, SRSRAN_MAX_CARRIERS> sched::get_enb_ue_cc_map(uint16_t rnti)
{
  std::array<int, SRSRAN_MAX_CARRIERS> ret{};
  ret.fill(-1); // -1 for inactive & non-existent carriers
  ue_db_access_locked(
      rnti,
      [this, &ret](sched_ue& ue) {
        for (size_t enb_cc_idx = 0; enb_cc_idx < carrier_schedulers.size(); ++enb_cc_idx) {
          const sched_ue_cell* cc_ue = ue.find_ue_carrier(enb_cc_idx);
          if (cc_ue != nullptr) {
            ret[enb_cc_idx] = cc_ue->get_ue_cc_idx();
          }
        }
      },
      __PRETTY_FUNCTION__);
  return ret;
}

std::array<int, SRSRAN_MAX_CARRIERS> sched::get_enb_ue_activ_cc_map(uint16_t rnti)
{
  std::array<int, SRSRAN_MAX_CARRIERS> ret{};
  ret.fill(-1); // -1 for inactive & non-existent carriers
  ue_db_access_locked(
      rnti,
      [this, &ret](sched_ue& ue) {
        for (size_t enb_cc_idx = 0; enb_cc_idx < carrier_schedulers.size(); ++enb_cc_idx) {
          const sched_ue_cell* cc_ue = ue.find_ue_carrier(enb_cc_idx);
          if (cc_ue != nullptr and (cc_ue->cc_state() == cc_st::active or cc_ue->cc_state() == cc_st::activating)) {
            ret[enb_cc_idx] = cc_ue->get_ue_cc_idx();
          }
        }
      },
      __PRETTY_FUNCTION__);
  return ret;
}

/*******************************************************
 *
 * Main sched functions
 *
 *******************************************************/

// Downlink Scheduler API
int sched::dl_sched(uint32_t tti_tx_dl, uint32_t enb_cc_idx, sched_interface::dl_sched_res_t& sched_result)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  if (not configured) {
    return 0;
  }
  if (enb_cc_idx >= carrier_schedulers.size()) {
    return 0;
  }

  tti_point tti_rx = tti_point{tti_tx_dl} - TX_ENB_DELAY;
  new_tti(tti_rx);

  // copy result
  sched_result = sched_results.get_sf(tti_rx)->get_cc(enb_cc_idx)->dl_sched_result;

  return 0;
}

// Uplink Scheduler API
int sched::ul_sched(uint32_t tti, uint32_t enb_cc_idx, srsenb::sched_interface::ul_sched_res_t& sched_result)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  if (not configured) {
    return 0;
  }
  if (enb_cc_idx >= carrier_schedulers.size()) {
    return 0;
  }

  // Compute scheduling Result for tti_rx
  tti_point tti_rx = tti_point{tti} - TX_ENB_DELAY - FDD_HARQ_DELAY_DL_MS;
  new_tti(tti_rx);

  // copy result
  sched_result = sched_results.get_sf(tti_rx)->get_cc(enb_cc_idx)->ul_sched_result;

  return SRSRAN_SUCCESS;
}

/// Generate scheduling decision for tti_rx, if it wasn't already generated
/// NOTE: The scheduling decision is made for all CCs in a single call/lock, otherwise the UE can have different
///       configurations (e.g. different set of activated SCells) in different CC decisions
void sched::new_tti(tti_point tti_rx)
{
  last_tti = std::max(last_tti, tti_rx);

  // Generate sched results for all CCs, if not yet generated
  for (size_t cc_idx = 0; cc_idx < carrier_schedulers.size(); ++cc_idx) {
    if (not is_generated(tti_rx, cc_idx)) {
      // Generate carrier scheduling result
      carrier_schedulers[cc_idx]->generate_tti_result(tti_rx);
    }
  }
}

/// Check if TTI result is generated
bool sched::is_generated(srsran::tti_point tti_rx, uint32_t enb_cc_idx) const
{
  return sched_results.has_sf(tti_rx) and sched_results.get_sf(tti_rx)->is_generated(enb_cc_idx);
}


// added in from srsRAN 4G updated repository
int sched::metrics_read(uint16_t rnti, mac_ue_metrics_t& metrics)
{
  return ue_db_access_locked(
      rnti, [&metrics](sched_ue& ue) { ue.metrics_read(metrics); }, "metrics_read");
}


// Common way to access ue_db elements in a read locking way
template <typename Func>
int sched::ue_db_access_locked(uint16_t rnti, Func&& f, const char* func_name, bool log_fail)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  auto                        it = ue_db.find(rnti);
  if (it != ue_db.end()) {
    f(*it->second);
  } else {
    if (log_fail) {
      if (func_name != nullptr) {
        Error("SCHED: User rnti=0x%x not found. Failed to call %s.", rnti, func_name);
      } else {
        Error("SCHED: User rnti=0x%x not found.", rnti);
      }
    }
    return SRSRAN_ERROR;
  }
  return SRSRAN_SUCCESS;
}

////////////////////////////////////
// E2 Agent Ctrl Slicing
// Setting the UEs slice information
////////////////////////////////////


std::vector<sched_ue*> sched::ues_in_slice(uint32_t slice_id)
{
  std::vector<sched_ue*> ues; 
  for(auto it = ue_db.begin(); it != ue_db.end(); ++it){
    sched_ue& ue = *it->second;  

    if(ue.slice_id() == (int64_t)slice_id){
      ues.push_back(&ue);
    }
  }
  return ues;
}

int cmpfunc_s(const void* a, const void* b)
{
  const fr_slice_t* slice_x = (fr_slice_t*) a;
  const fr_slice_t* slice_y = (fr_slice_t*) b;
  const int* x = (int*) &slice_x->id;
  const int* y = (int*) &slice_y->id;
  return (*x - *y);
}

slice_ctrl_out_e sched::slice_add_mod(slice_conf_t const& conf)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  Console("SLICE CTRL MSG: ADD SLICE\n");
  Slicing &slice_stats = Slicing::getInstance();
  ul_dl_slice_conf_t conf_dl = conf.dl;
  ul_dl_slice_conf_t* stats_dl = &slice_stats.stats_slice_conf.dl;
  ue_slice_conf_t* stats_ue_s = &slice_stats.stats_ue_slice_conf;

  // Save new sched algo
  // if len_slices = 0 (reset slice), we have to setup the default user scheduling algo
  std::string ssched_name = "";
  if (!strcmp(conf.dl.sched_name, "RR"))
    ssched_name = "time_rr";
  else if (!strcmp(conf.dl.sched_name, "PF"))
    ssched_name = "time_pf";
  else {
    Console("Unknown sched algo received, ssched_name %s\n", ssched_name.c_str());
    return SLICE_CTRL_OUT_ERROR;
  }
  char const* csched_name = ssched_name.c_str();
  stats_dl->len_sched_name = std::strlen(csched_name);
  stats_dl->sched_name = (char*)malloc(stats_dl->len_sched_name);
  srsran_assert(stats_dl->sched_name != NULL, "memory exhausted");
  memcpy(stats_dl->sched_name, csched_name, stats_dl->len_sched_name);

  // Check len_slices
  stats_dl->len_slices = conf_dl.len_slices;

  // TODO: len_slices = 0, reset slice as None

  if (stats_dl->len_slices > 0 && stats_dl->len_slices < 5) {
    stats_dl->slices = (fr_slice_t*)calloc(stats_dl->len_slices, sizeof(fr_slice_t));
    srsran_assert(stats_dl->slices != NULL, "memory exhausted");

    for (size_t i = 0; i < stats_dl->len_slices; ++i) {
      fr_slice_t const& conf_dl_s = conf_dl.slices[i];
      fr_slice_t* st_slice = &stats_dl->slices[i];

      // Check slice algo
      if (conf_dl_s.params.type != SLICE_ALG_SM_V0_STATIC) {
        Console("Not support algo = %d\n", conf_dl_s.params.type);
        return SLICE_CTRL_OUT_ERROR;
      }

      // Save slice algo data
      st_slice->params.type = conf_dl_s.params.type;
      static_slice_t conf_sta = conf_dl_s.params.u.sta;
      if (conf_sta.pos_low > 14 || conf_sta.pos_high > 14 || (conf_sta.pos_low > conf_sta.pos_high)) {
        Console("FAILED: SET DL SLICE ALGO %d, id %u, pos_low %u, pos_high %u\n",
                conf_dl_s.params.type, conf_dl_s.id, conf_sta.pos_low, conf_sta.pos_high);
        return SLICE_CTRL_OUT_ERROR;
      }
      static_slice_t* sta = &st_slice->params.u.sta;
      sta->pos_high = conf_sta.pos_high;
      sta->pos_low = conf_sta.pos_low;
      Console("SUCCESS: SET DL SLICE ALGO %d, id %u, pos_low %u, pos_high %u\n",
             conf_dl_s.params.type, conf_dl_s.id, sta->pos_low, sta->pos_high);

      // Save new id
      st_slice->id = conf_dl_s.id;
      // Associate all the ue to the first slice id
      if (i == 0)
        for (size_t j = 0; j < stats_ue_s->len_ue_slice; j++)
          stats_ue_s->ues[j].dl_id = st_slice->id;

      // Save new label
      st_slice->len_label = strlen(conf_dl_s.label);
      st_slice->label = (char*)malloc(st_slice->len_label);
      srsran_assert(st_slice->label != NULL, "memory exhausted");
      memcpy(st_slice->label, conf_dl_s.label, st_slice->len_label);

      // Save new sched algo
      std::string ssched = "";
      if (!strcmp(conf_dl_s.sched, "RR"))
        ssched = "time_rr";
      else if (!strcmp(conf_dl_s.sched, "PF"))
        ssched = "time_pf";
      else {
        Console("Unknown sched algo received, ssched %s\n", ssched.c_str());
        return SLICE_CTRL_OUT_ERROR;
      }
      char const* csched = ssched_name.c_str();
      st_slice->len_sched = std::strlen(csched);
      st_slice->sched = (char*)malloc(st_slice->len_sched);
      srsran_assert(st_slice->sched != NULL, "memory exhausted");
      memcpy(st_slice->sched, csched, st_slice->len_sched);
    }
    return SLICE_CTRL_OUT_OK;
  } else {
    Console("Not support len_slices = %u\n", stats_dl->len_slices);
    return SLICE_CTRL_OUT_ERROR;
  }
}

slice_ctrl_out_e sched::ue_slice_conf(ue_slice_conf_t const& ue_slice)
{
  std::lock_guard<std::mutex> lock(sched_mutex);
  Console("SLICE CTRL MSG: ASSOCIATE UE SLICE\n");
  Slicing &slice_stats = Slicing::getInstance();
  ul_dl_slice_conf_t* stats_dl = &slice_stats.stats_slice_conf.dl;
  ue_slice_conf_t* stats_ue_s = &slice_stats.stats_ue_slice_conf;

  if (stats_dl->len_slices == 0) {
    Console("No slice be added, UE can not be associated\n");
    return SLICE_CTRL_OUT_ERROR;
  }

  if (ue_db.size() <= 0) {
    Console("No UE connected\n");
    return SLICE_CTRL_OUT_ERROR;
  }

  // Get new ue slice config
  ue_slice_assoc_t* new_ues = ue_slice.ues;
  size_t n_new_ues = ue_slice.len_ue_slice;
  srsran_assert(n_new_ues != 0, "len_ue_slice == %u", n_new_ues);

  // Save new ue slice config
  for (size_t i = 0; i < n_new_ues; ++i) {
    uint16_t rnti = new_ues[i].rnti;

    // TODO: enable UL slicing
    if (new_ues[i].ul_id)
      Console("ignoring UL slice association for RNTI %04x\n", rnti);

    // Check rnti exists in ue_db
    auto new_it = ue_db.find(rnti);
    if (not ue_db.contains(rnti)) {
      Console("RNTI %04x doesn't exist in enb\n", rnti);
      return SLICE_CTRL_OUT_ERROR;
    }

    // Check new associated slice id exists in enb
    int new_idx = new_ues[i].dl_id;
    sched_ue& cur_ue = *new_it->second.get();
    int cur_idx = cur_ue.slice_id();
    if (new_idx == cur_idx) {
      Console("expected DL slice association for UE RNTI %04x\n", rnti);
      return SLICE_CTRL_OUT_ERROR;
    }
    // Get new associated slice pointer by binary search
    fr_slice_t* pslice = (fr_slice_t*) std::bsearch(&new_idx, stats_dl->slices, stats_dl->len_slices, sizeof(stats_dl->slices[0]), cmpfunc_s);
    if (pslice == NULL) {
      Console("dl_id %u doesn't exist\n", new_idx);
      return SLICE_CTRL_OUT_ERROR;
    }

    // TODO: Get ue pointer by binary search
    // ue_slice_assoc_t* pue = (ue_slice_assoc_t*) std::bsearch(&rnti, stats_ue_s->ues, stats_ue_s->len_ue_slice, sizeof(stats_ue_s->ues[0]), cmpfunc_u);

    // Get index of rnti in ue slice stats
    int idx_u = -1;
    for (size_t j = 0; j < stats_ue_s->len_ue_slice; j++) {
      if (stats_ue_s->ues[j].rnti == rnti) {
        idx_u = j;
        break;
      }
    }
    if (idx_u < 0) {
      Console("RNTI %04x doesn't exist in ue slice stats\n", rnti);
      return SLICE_CTRL_OUT_ERROR;
    }

    // Assoc ue to dl slice
    cur_ue.set_slice_id(new_idx);
    stats_ue_s->ues[idx_u].dl_id = new_idx;
    Console("SET UE rnti %x ASSOC DL ID %u\n", rnti, new_idx);
  }
  return SLICE_CTRL_OUT_OK;
}

slice_ctrl_out_e sched::slice(slice_ctrl_req_data_t const& s)
{

  if (s.msg.type == SLICE_CTRL_SM_V0_ADD) {
    return slice_add_mod(s.msg.u.add_mod_slice);
  } else if (s.msg.type == SLICE_CTRL_SM_V0_UE_SLICE_ASSOC) {
    return ue_slice_conf(s.msg.u.ue_slice);
  } else if (s.msg.type == SLICE_CTRL_SM_V0_DEL) {
    // TODO: enable delete slice
    Console("not support delete slice\n");
    return SLICE_CTRL_OUT_ERROR;
  } else {
    srsran_assert(0 != 0, "Unknow slice ctrl msg type %d", s.msg.type);
    return SLICE_CTRL_OUT_ERROR;
  }

}



} // namespace srsenb
