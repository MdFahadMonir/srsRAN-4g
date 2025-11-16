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

#include "srsenb/hdr/metrics_csv.h"
#include "srsran/phy/utils/vector.h"

#include <float.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>

#include <stdio.h>

using namespace std;

namespace srsenb {

metrics_csv::metrics_csv(std::string filename) : n_reports(0), metrics_report_period(1.0), enb(NULL)
{
  file.open(filename.c_str(), std::ios_base::out);
}

metrics_csv::~metrics_csv()
{
  stop();
}

void metrics_csv::set_handle(enb_metrics_interface* enb_)
{
  enb = enb_;
}

void metrics_csv::stop()
{
  if (file.is_open()) {
    file << "#eof\n";
    file.flush();
    file.close();
  }
}

void metrics_csv::set_metrics(const enb_metrics_t& metrics, const uint32_t period_usec)
{
  if (file.is_open() && enb != NULL) {
    if (n_reports == 0) {
      // file << "time;nof_ue;rnti;"
      //         "phy_pusch_sinr;phy_pucch_sinr;phy_rssi;phy_mcs;"
      //         "mac_dl_brate;mac_ul_brate;"
      //         "mac_tx_pkts;mac_tx_errors;mac_rx_pkts;mac_rx_errors;mac_ul_buffer;"
      //         "mac_dl_buffer;mac_dl_cqi;mac_dl_ri;mac_dl_rmi;mac_phr;"
      //         "rlc_num_tx_sdus;rlc_num_rx_sdus;rlc_num_tx_sdu_bytes;rlc_num_rx_sdu_bytes;"
      //         "rlc_num_lost_sdus;rlc_rx_latency_ms;rlc_num_tx_pdus;rlc_num_rx_pdus;"
      //         "rlc_num_tx_pdu_bytes;rlc_num_rx_pdu_bytes;rlc_num_lost_pdus;rlc_rx_buffered_bytes;"
      //         "pdcp_num_tx_pdus;pdcp_num_rx_pdus;pdcp_num_tx_pdu_bytes;pdcp_num_rx_pdu_bytes;"
      //         "pdcp_num_tx_acked_bytes;pdcp_tx_notification_latency_ms;pdcp_num_tx_buffered_pdus;"
      //         "pdcp_num_tx_buffered_pdus_bytes";
              // "proc_rmem;proc_rmem_kB;proc_vmem_kB;"
              // "sys_mem;system_load;thread_count";

      // file << "time,nof_ue,rnti,"
      //         "phy_pusch_sinr,phy_rssi,phy_mcs,"
      //         "mac_dl_brate,mac_ul_brate,mac_ul_buffer,mac_dl_buffer,"
      //         "mac_dl_cqi,mac_dl_ri,mac_dl_pmi";
      // file << "time,nof_ue,rnti,"
      //         "phy_pusch_sinr,phy_rssi,phy_mcs,"
      //         "mac_dl_brate,mac_ul_brate,mac_ul_buffer,mac_dl_buffer,"
      //         "mac_dl_cqi,mac_dl_ri,mac_dl_pmi,mac_n_prb";

      file << "time,nof_ue,rnti,"
              "phy_mcs,"
              "mac_dl_brate,mac_ul_brate,mac_ul_buffer,mac_dl_buffer,"
              "mac_dl_cqi,mac_dl_ri,mac_dl_pmi,mac_n_prb,pdcp";

      // // Add the cpus
      // for (uint32_t i = 0, e = metrics.sys.cpu_count; i != e; ++i) {
      //   file << ";cpu_" << std::to_string(i);
      // }

      // Add the new line.
      file << "\n";
    }


    // Sum up rates for all UEs
    float dl_rate_sum = 0.0, ul_rate_sum = 0.0;
    for (size_t i = 0; i < metrics.stack.rrc.ues.size(); i++) {
      dl_rate_sum = metrics.stack.mac.ues[i].tx_brate / (metrics.stack.mac.ues[i].nof_tti * 1e-3);
      ul_rate_sum = metrics.stack.mac.ues[i].rx_brate / (metrics.stack.mac.ues[i].nof_tti * 1e-3);
      float    dl_cqi = metrics.stack.mac.ues[i].dl_cqi;
      float    dl_ri = metrics.stack.mac.ues[i].dl_ri;
      float    dl_pmi = metrics.stack.mac.ues[i].dl_pmi;
      float    phr = metrics.stack.mac.ues[i].phr;
      float    pusch = metrics.phy[i].ul.pusch_sinr;
      float    pucch = metrics.phy[i].ul.pucch_sinr;
      float    rssi = metrics.phy[i].ul.rssi;
      float    mcs = metrics.phy[i].dl.mcs;
      float    pdcp_thr = metrics.stack.pdcp.ues[i].bearer[3].num_tx_pdu_bytes * 8 / (metrics.stack.mac.ues[i].nof_tti * 1e-3 * 1e6);;

      // std::cout << "ues size: " << metrics.stack.rrc.ues.size() << std::endl;
      // std::cout << "Logging dl_ri: " << dl_ri << std::endl;
      // printf ("%f\n", dl_ri);

      // allocated prbs
      float    alloc_prb = metrics.stack.mac.ues[i].allocated_prbs;

      // Time
      auto current = std::chrono::system_clock::now();
      auto seconds = std::chrono::duration_cast<std::chrono::milliseconds>(current.time_since_epoch());
      uint64_t time_s = seconds.count();

      // file << (metrics_report_period * n_reports) << ";";
      float dummy = metrics_report_period;
      file << time_s << ",";

      // UEs
      file << (metrics.stack.rrc.ues.size()) << ",";

      file << std::to_string(metrics.stack.mac.ues[i].rnti) << ",";

      // file << float_to_string(pusch, 3);

      // file << float_to_string(pucch, 3);

      // file << float_to_string(rssi, 3);

      if (mcs > 0) {
        file << float_to_string(SRSRAN_MAX(0.0, mcs), 3);
      } else {
        file << float_to_string(0, 2);
      }

      // DL rate
      if (dl_rate_sum > 0) {
        file << float_to_string(SRSRAN_MAX(0.1, (float)dl_rate_sum),3);
      } else {
        file << float_to_string(0, 2);
      }

      // UL rate
      if (ul_rate_sum > 0) {
        file << float_to_string(SRSRAN_MAX(0.1, (float)ul_rate_sum), 3);
      } else {
        file << float_to_string(0, 2);
      }

      // file << std::to_string(metrics.stack.mac.ues[i].tx_pkts) << ";";
      // file << std::to_string(metrics.stack.mac.ues[i].tx_errors) << ";";
      // file << std::to_string(metrics.stack.mac.ues[i].rx_pkts) << ";";
      // file << std::to_string(metrics.stack.mac.ues[i].rx_errors) << ";";
      file << std::to_string(metrics.stack.mac.ues[i].ul_buffer) << ",";
      file << std::to_string(metrics.stack.mac.ues[i].dl_buffer) << ",";
      // file << std::to_string(metrics.stack.mac.ues[i].ul_buffer) << ";";
      
      
      if (dl_cqi > 0) {
        file << float_to_string(SRSRAN_MAX(0.0, (float)dl_cqi), 3);
      } else {
        file << float_to_string(0, 2);
      }

      // static float previous_ri_value = 1; 
      // if (dl_ri == 0) {
      //   dl_ri = previous_ri_value;
      // } 
      // else {
      //   previous_ri_value = dl_ri;
      // }

      if (dl_ri > 0) {
        file << float_to_string(SRSRAN_MAX(0.0, (float)dl_ri), 3);
      } else {
        file << float_to_string(0, 2);
      }

      if (dl_pmi > 0) {
        // file << float_to_string(SRSRAN_MAX(0.0, (float)dl_pmi), 3);
        file << std::to_string(metrics.stack.mac.ues[i].dl_pmi) << ",";
      } else {
        file << float_to_string(0, 2);
      }


      // allocated prbs
      if (alloc_prb > 0) {
        file << float_to_string(SRSRAN_MAX(0.0, (float)alloc_prb), 3);
      } else {
        file << float_to_string(0, 2);
      }

      if (pdcp_thr > 0) {
        file << float_to_string(SRSRAN_MAX(0.0, (float)pdcp_thr), 3);
      } else {
        file << float_to_string(0, 2);
      }


      // if (phr > 0) {
      //   file << float_to_string(SRSRAN_MAX(0.0, (float)phr), 3);
      // } else {
      //   file << float_to_string(0, 2);
      // }

      // const srsran::rlc_bearer_metrics_t& src = metrics.stack.rlc.ues[i].bearer[3];
      // file << std::to_string(src.num_tx_sdus) << ",";
      // file << std::to_string(src.num_rx_sdus) << ",";
      // file << std::to_string(src.num_tx_sdu_bytes) << ",";
      // file << std::to_string(src.num_rx_sdu_bytes) << ",";
      // file << std::to_string(src.num_lost_sdus) << ",";
      // file << std::to_string(src.rx_latency_ms) << ",";
      // file << std::to_string(src.num_tx_pdus) << ",";
      // file << std::to_string(src.num_rx_pdus) << ",";
      // file << std::to_string(src.num_tx_pdu_bytes) << ",";
      // file << std::to_string(src.num_rx_pdu_bytes) << ",";
      // file << std::to_string(src.num_lost_pdus) << ",";
      // file << std::to_string(src.rx_buffered_bytes) << ",";

      // const srsran::pdcp_bearer_metrics_t& psrc = metrics.stack.pdcp.ues[i].bearer[3];
      // file << std::to_string(psrc.num_tx_pdus) << ",";
      // file << std::to_string(psrc.num_rx_pdus) << ",";
      // file << std::to_string(psrc.num_tx_pdu_bytes) << ",";
      // file << std::to_string(psrc.num_rx_pdu_bytes) << ",";
      // file << std::to_string(psrc.num_tx_acked_bytes) << ",";
      // file << std::to_string(psrc.tx_notification_latency_ms) << ",";
      // file << std::to_string(psrc.num_tx_buffered_pdus) << ",";
      // file << std::to_string(psrc.num_tx_buffered_pdus_bytes) << ",";  

      // // Write system metrics.
      // const srsran::sys_metrics_t& m = metrics.sys;
      // file << float_to_string(m.process_realmem, 2);
      // file << std::to_string(m.process_realmem_kB) << ";";
      // file << std::to_string(m.process_virtualmem_kB) << ";";
      // file << float_to_string(m.system_mem, 2);
      // file << float_to_string(m.process_cpu_usage, 2);
      // file << std::to_string(m.thread_count) << ";";

      // // Write the cpu metrics.
      // for (uint32_t i = 0, e = m.cpu_count, last_cpu_index = e - 1; i != e; ++i) {
      //   file << float_to_string(m.cpu_load[i], 2, (i != last_cpu_index));
      // }

    file << "\n";
}

    n_reports++;
  } else {
    std::cout << "Error, couldn't write CSV file." << std::endl;
  }
}

std::string metrics_csv::float_to_string(float f, int digits, bool add_semicolon)
{
  std::ostringstream os;
  const int          precision = (f == 0.0) ? digits - 1 : digits - log10f(fabs(f)) - 2 * DBL_EPSILON;
  os << std::fixed << std::setprecision(precision) << f;
  if (add_semicolon)
    os << ',';
  return os.str();
}

} // namespace srsenb