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

#include <functional>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "srsran/common/common_helper.h"
#include "srsran/common/config_file.h"
#include "srsran/common/crash_handler.h"
#include "srsran/common/tsan_options.h"
#include "srsran/srslog/event_trace.h"
#include "srsran/srslog/srslog.h"
#include "srsran/support/emergency_handlers.h"
#include "srsran/support/signal_handler.h"

#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <iostream>
#include <memory>
#include <srsran/common/string_helpers.h>
#include <string>

#include "srsenb/hdr/enb.h"
#include "srsenb/hdr/metrics_csv.h"
#include "srsenb/hdr/metrics_json.h"
#include "srsenb/hdr/metrics_stdout.h"
#include "srsran/common/enb_events.h"
#include "srsenb/hdr/stack/mac/sched_interface.h"

//////////////////////////////////
//// E2 Agent headers
//////////////////////////////////
  
#include "srsenb/sm/agent_if/read/sm_ag_if_rd.h"
#include "srsenb/sm/agent_if/write/sm_ag_if_wr.h"
#include "srsenb/sm/agent_if/ans/sm_ag_if_ans.h"
#include "srsenb/sm/sm_io.h"
#include "srsenb/e2_agent_api.h"
#include "srsenb/hdr/stack/enb_stack_lte.h"
//////////////////////////////////
//////////////////////////////////
//////////////////////////////////

using namespace std;
using namespace srsenb;
namespace bpo = boost::program_options;

/**********************************************************************
 *  Program arguments processing
 ***********************************************************************/
string                   config_file;
static bool              stdout_ts_enable = false;
static srslog::sink*     log_sink         = nullptr;
static std::atomic<bool> running          = {true};

void parse_args(all_args_t* args, int argc, char* argv[])
{
  string mcc;
  string mnc;
  string enb_id;
  bool   use_standard_lte_rates = false;

  // Command line only options
  bpo::options_description general("General options");
  // clang-format off
  general.add_options()
      ("help,h", "Produce help message")
      ("version,v", "Print version information and exit")
      ;

  // Command line or config file options
  bpo::options_description common("Configuration options");
  common.add_options()
    ("enb.enb_id",             bpo::value<string>(&enb_id)->default_value("0x0"),                       "eNodeB ID")
    ("enb.name",               bpo::value<string>(&args->stack.s1ap.enb_name)->default_value("srsenb01"), "eNodeB Name")
    ("enb.mcc",                bpo::value<string>(&mcc)->default_value("001"),                          "Mobile Country Code")
    ("enb.mnc",                bpo::value<string>(&mnc)->default_value("01"),                           "Mobile Network Code")
    ("enb.mme_addr",           bpo::value<string>(&args->stack.s1ap.mme_addr)->default_value("127.0.0.1"),"IP address of MME for S1 connection")
    ("enb.gtp_bind_addr",      bpo::value<string>(&args->stack.s1ap.gtp_bind_addr)->default_value("192.168.3.1"), "Local IP address to bind for GTP connection")
    ("enb.gtp_advertise_addr", bpo::value<string>(&args->stack.s1ap.gtp_advertise_addr)->default_value(""), "IP address of eNB to advertise for DL GTP-U Traffic")
    ("enb.s1c_bind_addr",      bpo::value<string>(&args->stack.s1ap.s1c_bind_addr)->default_value("192.168.3.1"), "Local IP address to bind for S1AP connection")
    ("enb.s1c_bind_port",      bpo::value<uint16_t>(&args->stack.s1ap.s1c_bind_port)->default_value(0), "Source port for S1AP connection (0 means any)")
    ("enb.n_prb",              bpo::value<uint32_t>(&args->enb.n_prb)->default_value(25),               "Number of PRB")
    ("enb.nof_ports",          bpo::value<uint32_t>(&args->enb.nof_ports)->default_value(1),            "Number of ports")
    ("enb.tm",                 bpo::value<uint32_t>(&args->enb.transmission_mode)->default_value(1),    "Transmission mode (1-8)")
    ("enb.p_a",                bpo::value<float>(&args->enb.p_a)->default_value(0.0f),                  "Power allocation rho_a (-6, -4.77, -3, -1.77, 0, 1, 2, 3)")

    ("enb_files.sib_config", bpo::value<string>(&args->enb_files.sib_config)->default_value("sib.conf"), "SIB configuration files")
    ("enb_files.rr_config",  bpo::value<string>(&args->enb_files.rr_config)->default_value("rr.conf"),   "RR configuration files")
    ("enb_files.rb_config", bpo::value<string>(&args->enb_files.rb_config)->default_value("rb.conf"), "SRB/DRB configuration files")

    ("rf.dl_earfcn",      bpo::value<uint32_t>(&args->enb.dl_earfcn)->default_value(0),   "Force Downlink EARFCN for single cell")
    ("rf.srate",          bpo::value<double>(&args->rf.srate_hz)->default_value(0.0),     "Force Tx and Rx sampling rate in Hz")
    ("rf.rx_gain",        bpo::value<float>(&args->rf.rx_gain)->default_value(50),        "Front-end receiver gain")
    ("rf.tx_gain",        bpo::value<float>(&args->rf.tx_gain)->default_value(70),        "Front-end transmitter gain")
    ("rf.tx_gain[0]",     bpo::value<float>(&args->rf.tx_gain_ch[0])->default_value(-1),  "Front-end transmitter gain CH0")
    ("rf.tx_gain[1]",     bpo::value<float>(&args->rf.tx_gain_ch[1])->default_value(-1),  "Front-end transmitter gain CH1")
    ("rf.tx_gain[2]",     bpo::value<float>(&args->rf.tx_gain_ch[2])->default_value(-1),  "Front-end transmitter gain CH2")
    ("rf.tx_gain[3]",     bpo::value<float>(&args->rf.tx_gain_ch[3])->default_value(-1),  "Front-end transmitter gain CH3")
    ("rf.tx_gain[4]",     bpo::value<float>(&args->rf.tx_gain_ch[4])->default_value(-1),  "Front-end transmitter gain CH4")
    ("rf.dl_freq",        bpo::value<float>(&args->rf.dl_freq)->default_value(-1),        "Downlink Frequency (if positive overrides EARFCN)")
    ("rf.ul_freq",        bpo::value<float>(&args->rf.ul_freq)->default_value(-1),        "Uplink Frequency (if positive overrides EARFCN)")

    ("rf.device_name",       bpo::value<string>(&args->rf.device_name)->default_value("auto"),       "Front-end device name")
    ("rf.device_args",       bpo::value<string>(&args->rf.device_args)->default_value("auto"),       "Front-end device arguments")
    ("rf.time_adv_nsamples", bpo::value<string>(&args->rf.time_adv_nsamples)->default_value("auto"), "Transmission time advance")

    ("gui.enable",        bpo::value<bool>(&args->gui.enable)->default_value(false),          "Enable GUI plots")

    /* Log section */
    ("log.rf_level",     bpo::value<string>(&args->rf.log_level),         "RF log level")
    ("log.phy_level",     bpo::value<string>(&args->phy.log.phy_level),   "PHY log level")
    ("log.phy_hex_limit", bpo::value<int>(&args->phy.log.phy_hex_limit),  "PHY log hex dump limit")
    ("log.phy_lib_level", bpo::value<string>(&args->phy.log.phy_lib_level)->default_value("none"), "PHY lib log level")
    ("log.mac_level",     bpo::value<string>(&args->stack.log.mac_level),   "MAC log level")
    ("log.mac_hex_limit", bpo::value<int>(&args->stack.log.mac_hex_limit),  "MAC log hex dump limit")
    ("log.rlc_level",     bpo::value<string>(&args->stack.log.rlc_level),   "RLC log level")
    ("log.rlc_hex_limit", bpo::value<int>(&args->stack.log.rlc_hex_limit),  "RLC log hex dump limit")
    ("log.pdcp_level",    bpo::value<string>(&args->stack.log.pdcp_level),  "PDCP log level")
    ("log.pdcp_hex_limit",bpo::value<int>(&args->stack.log.pdcp_hex_limit), "PDCP log hex dump limit")
    ("log.rrc_level",     bpo::value<string>(&args->stack.log.rrc_level),   "RRC log level")
    ("log.rrc_hex_limit", bpo::value<int>(&args->stack.log.rrc_hex_limit),  "RRC log hex dump limit")
    ("log.gtpu_level",    bpo::value<string>(&args->stack.log.gtpu_level),  "GTPU log level")
    ("log.gtpu_hex_limit",bpo::value<int>(&args->stack.log.gtpu_hex_limit), "GTPU log hex dump limit")
    ("log.s1ap_level",    bpo::value<string>(&args->stack.log.s1ap_level),  "S1AP log level")
    ("log.s1ap_hex_limit",bpo::value<int>(&args->stack.log.s1ap_hex_limit), "S1AP log hex dump limit")
    ("log.stack_level",    bpo::value<string>(&args->stack.log.stack_level),  "Stack log level")
    ("log.stack_hex_limit",bpo::value<int>(&args->stack.log.stack_hex_limit), "Stack log hex dump limit")

    ("log.all_level",     bpo::value<string>(&args->log.all_level)->default_value("info"),   "ALL log level")
    ("log.all_hex_limit", bpo::value<int>(&args->log.all_hex_limit)->default_value(32),  "ALL log hex dump limit")

    ("log.filename",      bpo::value<string>(&args->log.filename)->default_value("/tmp/ue.log"),"Log filename")
    ("log.file_max_size", bpo::value<int>(&args->log.file_max_size)->default_value(-1), "Maximum file size (in kilobytes). When passed, multiple files are created. Default -1 (single file)")

    /* PCAP */
    ("pcap.enable",    bpo::value<bool>(&args->stack.mac_pcap.enable)->default_value(false),         "Enable MAC packet captures for wireshark")
    ("pcap.filename",  bpo::value<string>(&args->stack.mac_pcap.filename)->default_value("enb_mac.pcap"), "MAC layer capture filename")
    ("pcap.nr_filename",  bpo::value<string>(&args->nr_stack.mac.pcap.filename)->default_value("enb_mac_nr.pcap"), "NR MAC layer capture filename")
    ("pcap.s1ap_enable",   bpo::value<bool>(&args->stack.s1ap_pcap.enable)->default_value(false),         "Enable S1AP packet captures for wireshark")
    ("pcap.s1ap_filename", bpo::value<string>(&args->stack.s1ap_pcap.filename)->default_value("enb_s1ap.pcap"), "S1AP layer capture filename")
    ("pcap.mac_net_enable", bpo::value<bool>(&args->stack.mac_pcap_net.enable)->default_value(false),         "Enable MAC network captures")
    ("pcap.bind_ip", bpo::value<string>(&args->stack.mac_pcap_net.bind_ip)->default_value("0.0.0.0"),         "Bind IP address for MAC network trace")
    ("pcap.bind_port", bpo::value<uint16_t>(&args->stack.mac_pcap_net.bind_port)->default_value(5687),        "Bind port for MAC network trace")
    ("pcap.client_ip", bpo::value<string>(&args->stack.mac_pcap_net.client_ip)->default_value("127.0.0.1"),     "Client IP address for MAC network trace")
    ("pcap.client_port", bpo::value<uint16_t>(&args->stack.mac_pcap_net.client_port)->default_value(5847),    "Enable MAC network captures")

    /* Scheduling section */
    ("scheduler.policy", bpo::value<string>(&args->stack.mac.sched.sched_policy)->default_value("time_pf"), "DL and UL data scheduling policy (E.g. time_rr, time_pf)")
    ("scheduler.policy_args", bpo::value<string>(&args->stack.mac.sched.sched_policy_args)->default_value("2"), "Scheduler policy-specific arguments")
    ("scheduler.pdsch_mcs", bpo::value<int>(&args->stack.mac.sched.pdsch_mcs)->default_value(-1), "Optional fixed PDSCH MCS (ignores reported CQIs if specified)")
    ("scheduler.pdsch_max_mcs", bpo::value<int>(&args->stack.mac.sched.pdsch_max_mcs)->default_value(-1), "Optional PDSCH MCS limit")
    ("scheduler.pusch_mcs", bpo::value<int>(&args->stack.mac.sched.pusch_mcs)->default_value(-1), "Optional fixed PUSCH MCS (ignores reported CQIs if specified)")
    ("scheduler.pusch_max_mcs", bpo::value<int>(&args->stack.mac.sched.pusch_max_mcs)->default_value(-1), "Optional PUSCH MCS limit")
    ("scheduler.min_aggr_level", bpo::value<int>(&args->stack.mac.sched.min_aggr_level)->default_value(0), "Optional minimum aggregation level index (l=log2(L)) ")
    ("scheduler.max_aggr_level", bpo::value<int>(&args->stack.mac.sched.max_aggr_level)->default_value(3), "Optional maximum aggregation level index (l=log2(L)) ")
    ("scheduler.adaptive_aggr_level", bpo::value<bool>(&args->stack.mac.sched.adaptive_aggr_level)->default_value(false), "Boolean flag to enable/disable adaptive aggregation level based on target BLER")
    ("scheduler.max_nof_ctrl_symbols", bpo::value<uint32_t>(&args->stack.mac.sched.max_nof_ctrl_symbols)->default_value(3), "Number of control symbols")
    ("scheduler.min_nof_ctrl_symbols", bpo::value<uint32_t>(&args->stack.mac.sched.min_nof_ctrl_symbols)->default_value(1), "Minimum number of control symbols")
    ("scheduler.pucch_multiplex_enable", bpo::value<bool>(&args->stack.mac.sched.pucch_mux_enabled)->default_value(false), "Enable PUCCH multiplexing")
    ("scheduler.pucch_harq_max_rb", bpo::value<int>(&args->stack.mac.sched.pucch_harq_max_rb)->default_value(0), "Maximum number of RB to be used for PUCCH on the edges of the grid")
    ("scheduler.target_bler", bpo::value<float>(&args->stack.mac.sched.target_bler)->default_value(0.05), "Target BLER (in decimal) to achieve via adaptive link")
    ("scheduler.max_delta_dl_cqi", bpo::value<float>(&args->stack.mac.sched.max_delta_dl_cqi)->default_value(5.0), "Maximum shift in CQI for adaptive DL link")
    ("scheduler.max_delta_ul_snr", bpo::value<float>(&args->stack.mac.sched.max_delta_ul_snr)->default_value(5.0), "Maximum shift in UL SNR for adaptive UL link")
    ("scheduler.adaptive_dl_mcs_step_size", bpo::value<float>(&args->stack.mac.sched.adaptive_dl_mcs_step_size)->default_value(0.001), "Step size or learning rate used in adaptive DL MCS link")
    ("scheduler.adaptive_ul_mcs_step_size", bpo::value<float>(&args->stack.mac.sched.adaptive_ul_mcs_step_size)->default_value(0.001), "Step size or learning rate used in adaptive UL MCS link")
    ("scheduler.min_tpc_tti_interval", bpo::value<uint32_t>(&args->stack.mac.sched.min_tpc_tti_interval)->default_value(1), "Minimum TTI interval between positive or negative TPCs")
    ("scheduler.ul_snr_avg_alpha", bpo::value<float>(&args->stack.mac.sched.ul_snr_avg_alpha)->default_value(0.05), "Exponential Average alpha coefficient used in estimation of UL SNR")
    ("scheduler.init_ul_snr_value", bpo::value<int>(&args->stack.mac.sched.init_ul_snr_value)->default_value(5), "Initial UL SNR value used for computing MCS in the first UL grant")
    ("scheduler.init_dl_cqi", bpo::value<int>(&args->stack.mac.sched.init_dl_cqi)->default_value(5), "DL CQI value used before any CQI report is available to the eNB")
    ("scheduler.max_sib_coderate", bpo::value<float>(&args->stack.mac.sched.max_sib_coderate)->default_value(0.8), "Upper bound on SIB and RAR grants coderate")
    ("scheduler.pdcch_cqi_offset", bpo::value<int>(&args->stack.mac.sched.pdcch_cqi_offset)->default_value(0), "CQI offset in derivation of PDCCH aggregation level")



    /* Downlink Channel emulator section */
    ("channel.dl.enable",            bpo::value<bool>(&args->phy.dl_channel_args.enable)->default_value(false),               "Enable/Disable internal Downlink channel emulator")
    ("channel.dl.awgn.enable",       bpo::value<bool>(&args->phy.dl_channel_args.awgn_enable)->default_value(false),          "Enable/Disable AWGN simulator")
    ("channel.dl.awgn.snr",          bpo::value<float>(&args->phy.dl_channel_args.awgn_snr_dB)->default_value(30.0f),         "Target SNR in dB")
    ("channel.dl.fading.enable",     bpo::value<bool>(&args->phy.dl_channel_args.fading_enable)->default_value(false),        "Enable/Disable Fading model")
    ("channel.dl.fading.model",      bpo::value<string>(&args->phy.dl_channel_args.fading_model)->default_value("none"),      "Fading model + maximum doppler (E.g. none, epa5, eva70, etu300, etc)")
    ("channel.dl.delay.enable",      bpo::value<bool>(&args->phy.dl_channel_args.delay_enable)->default_value(false),         "Enable/Disable Delay simulator")
    ("channel.dl.delay.period_s",    bpo::value<float>(&args->phy.dl_channel_args.delay_period_s)->default_value(3600),       "Delay period in seconds (integer)")
    ("channel.dl.delay.init_time_s", bpo::value<float>(&args->phy.dl_channel_args.delay_init_time_s)->default_value(0),       "Initial time in seconds")
    ("channel.dl.delay.maximum_us",  bpo::value<float>(&args->phy.dl_channel_args.delay_max_us)->default_value(100.0f),       "Maximum delay in microseconds")
    ("channel.dl.delay.minimum_us",  bpo::value<float>(&args->phy.dl_channel_args.delay_min_us)->default_value(10.0f),        "Minimum delay in microseconds")
    ("channel.dl.rlf.enable",        bpo::value<bool>(&args->phy.dl_channel_args.rlf_enable)->default_value(false),           "Enable/Disable Radio-Link Failure simulator")
    ("channel.dl.rlf.t_on_ms",       bpo::value<uint32_t >(&args->phy.dl_channel_args.rlf_t_on_ms)->default_value(10000),     "Time for On state of the channel (ms)")
    ("channel.dl.rlf.t_off_ms",      bpo::value<uint32_t >(&args->phy.dl_channel_args.rlf_t_off_ms)->default_value(2000),     "Time for Off state of the channel (ms)")
    ("channel.dl.hst.enable",        bpo::value<bool>(&args->phy.dl_channel_args.hst_enable)->default_value(false),           "Enable/Disable HST simulator")
    ("channel.dl.hst.period_s",      bpo::value<float>(&args->phy.dl_channel_args.hst_period_s)->default_value(7.2f),         "HST simulation period in seconds")
    ("channel.dl.hst.fd_hz",         bpo::value<float>(&args->phy.dl_channel_args.hst_fd_hz)->default_value(+750.0f),         "Doppler frequency in Hz")
    ("channel.dl.hst.init_time_s",   bpo::value<float>(&args->phy.dl_channel_args.hst_init_time_s)->default_value(0),         "Initial time in seconds")

    /* Uplink Channel emulator section */
    ("channel.ul.enable",            bpo::value<bool>(&args->phy.ul_channel_args.enable)->default_value(false),                  "Enable/Disable internal Downlink channel emulator")
    ("channel.ul.awgn.enable",       bpo::value<bool>(&args->phy.ul_channel_args.awgn_enable)->default_value(false),             "Enable/Disable AWGN simulator")
    ("channel.ul.awgn.signal_power", bpo::value<float>(&args->phy.ul_channel_args.awgn_signal_power_dBfs)->default_value(30.0f), "Received signal power in decibels full scale (dBfs)")
    ("channel.ul.awgn.snr",          bpo::value<float>(&args->phy.ul_channel_args.awgn_snr_dB)->default_value(30.0f),            "Noise level in decibels full scale (dBfs)")
    ("channel.ul.fading.enable",     bpo::value<bool>(&args->phy.ul_channel_args.fading_enable)->default_value(false),           "Enable/Disable Fading model")
    ("channel.ul.fading.model",      bpo::value<string>(&args->phy.ul_channel_args.fading_model)->default_value("none"),         "Fading model + maximum doppler (E.g. none, epa5, eva70, etu300, etc)")
    ("channel.ul.delay.enable",      bpo::value<bool>(&args->phy.ul_channel_args.delay_enable)->default_value(false),            "Enable/Disable Delay simulator")
    ("channel.ul.delay.period_s",    bpo::value<float>(&args->phy.ul_channel_args.delay_period_s)->default_value(3600),          "Delay period in seconds (integer)")
    ("channel.ul.delay.init_time_s", bpo::value<float>(&args->phy.ul_channel_args.delay_init_time_s)->default_value(0),          "Initial time in seconds")
    ("channel.ul.delay.maximum_us",  bpo::value<float>(&args->phy.ul_channel_args.delay_max_us)->default_value(100.0f),          "Maximum delay in microseconds")
    ("channel.ul.delay.minimum_us",  bpo::value<float>(&args->phy.ul_channel_args.delay_min_us)->default_value(10.0f),           "Minimum delay in microseconds")
    ("channel.ul.rlf.enable",        bpo::value<bool>(&args->phy.ul_channel_args.rlf_enable)->default_value(false),              "Enable/Disable Radio-Link Failure simulator")
    ("channel.ul.rlf.t_on_ms",       bpo::value<uint32_t >(&args->phy.ul_channel_args.rlf_t_on_ms)->default_value(10000),        "Time for On state of the channel (ms)")
    ("channel.ul.rlf.t_off_ms",      bpo::value<uint32_t >(&args->phy.ul_channel_args.rlf_t_off_ms)->default_value(2000),        "Time for Off state of the channel (ms)")
    ("channel.ul.hst.enable",        bpo::value<bool>(&args->phy.ul_channel_args.hst_enable)->default_value(false),              "Enable/Disable HST simulator")
    ("channel.ul.hst.period_s",      bpo::value<float>(&args->phy.ul_channel_args.hst_period_s)->default_value(7.2f),            "HST simulation period in seconds")
    ("channel.ul.hst.fd_hz",         bpo::value<float>(&args->phy.ul_channel_args.hst_fd_hz)->default_value(+750.0f),            "Doppler frequency in Hz")
    ("channel.ul.hst.init_time_s",   bpo::value<float>(&args->phy.ul_channel_args.hst_init_time_s)->default_value(0),            "Initial time in seconds")

      /* Expert section */
    ("expert.metrics_period_secs", bpo::value<float>(&args->general.metrics_period_secs)->default_value(1.0), "Periodicity for metrics in seconds.")
    ("expert.metrics_csv_enable",  bpo::value<bool>(&args->general.metrics_csv_enable)->default_value(false), "Write metrics to CSV file.")
    ("expert.metrics_csv_filename", bpo::value<string>(&args->general.metrics_csv_filename)->default_value("/tmp/enb_metrics.csv"), "Metrics CSV filename.")
    ("expert.pusch_max_its", bpo::value<uint32_t>(&args->phy.pusch_max_its)->default_value(8), "Maximum number of turbo decoder iterations for LTE.")
    ("expert.pusch_8bit_decoder", bpo::value<bool>(&args->phy.pusch_8bit_decoder)->default_value(false), "Use 8-bit for LLR representation and turbo decoder trellis computation (Experimental).")
    ("expert.pusch_meas_evm", bpo::value<bool>(&args->phy.pusch_meas_evm)->default_value(false), "Enable/Disable PUSCH EVM measure.")
    ("expert.tx_amplitude", bpo::value<float>(&args->phy.tx_amplitude)->default_value(0.6), "Transmit amplitude factor.")
    ("expert.nof_phy_threads", bpo::value<uint32_t>(&args->phy.nof_phy_threads)->default_value(3), "Number of PHY threads.")
    ("expert.nof_prach_threads", bpo::value<uint32_t>(&args->phy.nof_prach_threads)->default_value(1), "Number of PRACH workers per carrier. Only 1 or 0 is supported.")
    ("expert.max_prach_offset_us", bpo::value<float>(&args->phy.max_prach_offset_us)->default_value(30), "Maximum allowed RACH offset (in us).")
    ("expert.equalizer_mode", bpo::value<string>(&args->phy.equalizer_mode)->default_value("mmse"), "Equalizer mode.")
    ("expert.estimator_fil_w", bpo::value<float>(&args->phy.estimator_fil_w)->default_value(0.1), "Chooses the coefficients for the 3-tap channel estimator centered filter.")
    ("expert.lte_sample_rates", bpo::value<bool>(&use_standard_lte_rates)->default_value(false), "Whether to use default LTE sample rates instead of shorter variants.")
    ("expert.report_json_enable",  bpo::value<bool>(&args->general.report_json_enable)->default_value(false), "Write eNB report to JSON file (default disabled).")
    ("expert.report_json_filename", bpo::value<string>(&args->general.report_json_filename)->default_value("/tmp/enb_report.json"), "Report JSON filename (default /tmp/enb_report.json).")
    ("expert.report_json_asn1_oct",  bpo::value<bool>(&args->general.report_json_asn1_oct)->default_value(false), "Prints ASN1 messages encoded as an octet string instead of plain text in the JSON report file.")
    ("expert.alarms_log_enable",  bpo::value<bool>(&args->general.alarms_log_enable)->default_value(false), "Enable Alarms logging (default diabled).")
    ("expert.alarms_filename", bpo::value<string>(&args->general.alarms_filename)->default_value("/tmp/enb_alarms.log"), "Alarms logging filename (default /tmp/alarms.log).")
    ("expert.tracing_enable",  bpo::value<bool>(&args->general.tracing_enable)->default_value(false), "Events tracing.")
    ("expert.tracing_filename", bpo::value<string>(&args->general.tracing_filename)->default_value("/tmp/enb_tracing.log"), "Tracing events filename.")
    ("expert.tracing_buffcapacity", bpo::value<std::size_t>(&args->general.tracing_buffcapacity)->default_value(1000000), "Tracing buffer capcity.")
    ("expert.stdout_ts_enable", bpo::value<bool>(&stdout_ts_enable)->default_value(false), "Prints once per second the timestamp into stdout.")
    ("expert.rrc_inactivity_timer", bpo::value<uint32_t>(&args->general.rrc_inactivity_timer)->default_value(30000), "Inactivity timer in ms.")
    ("expert.print_buffer_state", bpo::value<bool>(&args->general.print_buffer_state)->default_value(false), "Prints on the console the buffer state every 10 seconds.")
    ("expert.eea_pref_list", bpo::value<string>(&args->general.eea_pref_list)->default_value("EEA0, EEA2, EEA1"), "Ordered preference list for the selection of encryption algorithm (EEA) (default: EEA0, EEA2, EEA1).")
    ("expert.eia_pref_list", bpo::value<string>(&args->general.eia_pref_list)->default_value("EIA2, EIA1, EIA0"), "Ordered preference list for the selection of integrity algorithm (EIA) (default: EIA2, EIA1, EIA0).")
    ("expert.nof_prealloc_ues", bpo::value<uint32_t>(&args->stack.mac.nof_prealloc_ues)->default_value(8), "Number of UE resources to preallocate during eNB initialization.")
    ("expert.lcid_padding", bpo::value<int>(&args->stack.mac.lcid_padding)->default_value(3), "LCID on which to put MAC padding")
    ("expert.max_mac_dl_kos", bpo::value<uint32_t>(&args->general.max_mac_dl_kos)->default_value(100), "Maximum number of consecutive KOs in DL before triggering the UE's release (default 100).")
    ("expert.max_mac_ul_kos", bpo::value<uint32_t>(&args->general.max_mac_ul_kos)->default_value(100), "Maximum number of consecutive KOs in UL before triggering the UE's release (default 100).")
    ("expert.gtpu_tunnel_timeout", bpo::value<uint32_t>(&args->stack.gtpu_indirect_tunnel_timeout_msec)->default_value(0), "Maximum time that GTPU takes to release indirect forwarding tunnel since the last received GTPU PDU (0 for infinity).")
    ("expert.rlf_release_timer_ms", bpo::value<uint32_t>(&args->general.rlf_release_timer_ms)->default_value(4000), "Time taken by eNB to release UE context after it detects an RLF.")
    ("expert.extended_cp", bpo::value<bool>(&args->phy.extended_cp)->default_value(false), "Use extended cyclic prefix")
    ("expert.ts1_reloc_prep_timeout", bpo::value<uint32_t>(&args->stack.s1ap.ts1_reloc_prep_timeout)->default_value(10000), "S1AP TS 36.413 TS1RelocPrep Expiry Timeout value in milliseconds.")
    ("expert.ts1_reloc_overall_timeout", bpo::value<uint32_t>(&args->stack.s1ap.ts1_reloc_overall_timeout)->default_value(10000), "S1AP TS 36.413 TS1RelocOverall Expiry Timeout value in milliseconds.")
    ("expert.rlf_min_ul_snr_estim", bpo::value<int>(&args->stack.mac.rlf_min_ul_snr_estim)->default_value(-2), "SNR threshold in dB below which the eNB is notified with rlf ko.")

    // eMBMS section
    ("embms.enable", bpo::value<bool>(&args->stack.embms.enable)->default_value(false), "Enables MBMS in the eNB")
    ("embms.m1u_multiaddr", bpo::value<string>(&args->stack.embms.m1u_multiaddr)->default_value("239.255.0.1"), "M1-U Multicast address the eNB joins.")
    ("embms.m1u_if_addr", bpo::value<string>(&args->stack.embms.m1u_if_addr)->default_value("127.0.1.201"), "IP address of the interface the eNB will listen for M1-U traffic.")
    ("embms.mcs", bpo::value<uint16_t>(&args->stack.embms.mcs)->default_value(20), "Modulation and Coding scheme of MBMS traffic.")

    // NR section
    ("scheduler.nr_pdsch_mcs", bpo::value<int>(&args->nr_stack.mac.sched_cfg.fixed_dl_mcs)->default_value(28), "Fixed NR DL MCS (-1 for dynamic).")
    ("scheduler.nr_pusch_mcs", bpo::value<int>(&args->nr_stack.mac.sched_cfg.fixed_ul_mcs)->default_value(28), "Fixed NR UL MCS (-1 for dynamic).")
    ("expert.nr_pusch_max_its", bpo::value<uint32_t>(&args->phy.nr_pusch_max_its)->default_value(10),     "Maximum number of LDPC iterations for NR.")

    // VNF params
    ("vnf.type", bpo::value<string>(&args->phy.vnf_args.type)->default_value("gnb"), "VNF instance type [gnb,ue].")
    ("vnf.addr", bpo::value<string>(&args->phy.vnf_args.bind_addr)->default_value("localhost"), "Address to bind VNF interface.")
    ("vnf.port", bpo::value<uint16_t>(&args->phy.vnf_args.bind_port)->default_value(3333), "Bind port.")
    ("log.vnf_level",     bpo::value<string>(&args->phy.vnf_args.log_level),   "VNF log level.")
    ("log.vnf_hex_limit", bpo::value<int>(&args->phy.vnf_args.log_hex_limit),  "VNF log hex dump limit.")
  ;

  // Positional options - config file location
  bpo::options_description position("Positional options");
  position.add_options()
    ("config_file", bpo::value< string >(&config_file), "eNodeB configuration file")
  ;

  // clang-format on
  bpo::positional_options_description p;
  p.add("config_file", -1);

  // these options are allowed on the command line
  bpo::options_description cmdline_options;
  cmdline_options.add(common).add(position).add(general);

  // parse the command line and store result in vm
  bpo::variables_map vm;
  try {
    bpo::store(bpo::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
    bpo::notify(vm);
  } catch (bpo::error& e) {
    cerr << e.what() << endl;
    exit(1);
  }
  // help option was given - print usage and exit
  if (vm.count("help")) {
    cout << "Usage: " << argv[0] << " [OPTIONS] config_file" << endl << endl;
    cout << common << endl << general << endl;
    exit(0);
  }

  // print version number and exit
  if (vm.count("version")) {
    cout << "Version " << srsran_get_version_major() << "." << srsran_get_version_minor() << "."
         << srsran_get_version_patch() << endl;
    exit(0);
  }

  // if no config file given, check users home path
  if (!vm.count("config_file")) {
    if (!config_exists(config_file, "enb.conf")) {
      cout << "Failed to read eNB configuration file " << config_file << " - exiting" << endl;
      exit(1);
    }
  }

  cout << "Reading configuration file " << config_file << "..." << endl;
  ifstream conf(config_file.c_str(), ios::in);
  if (conf.fail()) {
    cout << "Failed to read configuration file " << config_file << " - exiting" << endl;
    exit(1);
  }

  // parse config file and handle errors gracefully
  try {
    bpo::store(bpo::parse_config_file(conf, common), vm);
    bpo::notify(vm);
  } catch (const boost::program_options::error& e) {
    cerr << e.what() << endl;
    exit(1);
  }

  // Convert MCC/MNC strings
  if (!srsran::string_to_mcc(mcc, &args->stack.s1ap.mcc)) {
    cout << "Error parsing enb.mcc:" << mcc << " - must be a 3-digit string." << endl;
  }
  if (!srsran::string_to_mnc(mnc, &args->stack.s1ap.mnc)) {
    cout << "Error parsing enb.mnc:" << mnc << " - must be a 2 or 3-digit string." << endl;
  }

  if (args->stack.embms.enable) {
    if (args->stack.mac.sched.max_nof_ctrl_symbols == 3) {
      fprintf(stderr,
              "nof_ctrl_symbols = %d, While using MBMS, please set number of control symbols to either 1 or 2, "
              "depending on the length of the non-mbsfn region\n",
              args->stack.mac.sched.max_nof_ctrl_symbols);
      exit(1);
    }
  }

  // Check PRACH workers
  if (args->phy.nof_prach_threads > 1) {
    fprintf(stderr,
            "nof_prach_workers = %d. Value is not supported, only 0 or 1 are allowed\n",
            args->phy.nof_prach_threads);
    exit(1);
  }

  // Convert eNB Id
  std::size_t pos = {};
  try {
    args->enb.enb_id = std::stoi(enb_id, &pos, 0);
  } catch (...) {
    cout << "Error parsing enb.enb_id: " << enb_id << "." << endl;
    exit(1);
  }
  if (pos != enb_id.size()) {
    cout << "Error parsing enb.enb_id: " << enb_id << "." << endl;
    exit(1);
  }

  // Apply all_level to any unset layers
  if (vm.count("log.all_level")) {
    if (!vm.count("log.rf_level")) {
      args->rf.log_level = args->log.all_level;
    }
    if (!vm.count("log.phy_level")) {
      args->phy.log.phy_level = args->log.all_level;
    }
    if (!vm.count("log.phy_lib_level")) {
      args->phy.log.phy_lib_level = args->log.all_level;
    }
    if (!vm.count("log.mac_level")) {
      args->stack.log.mac_level = args->log.all_level;
    }
    if (!vm.count("log.rlc_level")) {
      args->stack.log.rlc_level = args->log.all_level;
    }
    if (!vm.count("log.pdcp_level")) {
      args->stack.log.pdcp_level = args->log.all_level;
    }
    if (!vm.count("log.rrc_level")) {
      args->stack.log.rrc_level = args->log.all_level;
    }
    if (!vm.count("log.gtpu_level")) {
      args->stack.log.gtpu_level = args->log.all_level;
    }
    if (!vm.count("log.s1ap_level")) {
      args->stack.log.s1ap_level = args->log.all_level;
    }
    if (!vm.count("log.stack_level")) {
      args->stack.log.stack_level = args->log.all_level;
    }
  }

  // Apply all_hex_limit to any unset layers
  if (vm.count("log.all_hex_limit")) {
    if (!vm.count("log.phy_hex_limit")) {
      args->log.phy_hex_limit = args->log.all_hex_limit;
    }
    if (!vm.count("log.mac_hex_limit")) {
      args->stack.log.mac_hex_limit = args->log.all_hex_limit;
    }
    if (!vm.count("log.rlc_hex_limit")) {
      args->stack.log.rlc_hex_limit = args->log.all_hex_limit;
    }
    if (!vm.count("log.pdcp_hex_limit")) {
      args->stack.log.pdcp_hex_limit = args->log.all_hex_limit;
    }
    if (!vm.count("log.rrc_hex_limit")) {
      args->stack.log.rrc_hex_limit = args->log.all_hex_limit;
    }
    if (!vm.count("log.gtpu_hex_limit")) {
      args->stack.log.gtpu_hex_limit = args->log.all_hex_limit;
    }
    if (!vm.count("log.s1ap_hex_limit")) {
      args->stack.log.s1ap_hex_limit = args->log.all_hex_limit;
    }
    if (!vm.count("log.stack_hex_limit")) {
      args->stack.log.stack_hex_limit = args->log.all_hex_limit;
    }
  }

  // Check remaining eNB config files
  if (!config_exists(args->enb_files.sib_config, "sib.conf")) {
    cout << "Failed to read SIB configuration file " << args->enb_files.sib_config << " - exiting" << endl;
    exit(1);
  }

  if (!config_exists(args->enb_files.rr_config, "rr.conf")) {
    cout << "Failed to read RR configuration file " << args->enb_files.rr_config << " - exiting" << endl;
    exit(1);
  }

  if (!config_exists(args->enb_files.rb_config, "rb.conf")) {
    cout << "Failed to read DRB configuration file " << args->enb_files.rb_config << " - exiting" << endl;
    exit(1);
  }

  srsran_use_standard_symbol_size(use_standard_lte_rates);
}

static bool do_metrics = false;
static bool do_padding = false;

static void execute_cmd(metrics_stdout* metrics, srsenb::enb_command_interface* control, const string& cmd_line)
{
  vector<string> cmd;
  srsran::string_parse_list(cmd_line, ' ', cmd);
  if (cmd[0] == "t") {
    do_metrics = !do_metrics;
    if (do_metrics) {
      cout << "Enter t to stop trace." << endl;
    } else {
      cout << "Enter t to restart trace." << endl;
    }
    metrics->toggle_print(do_metrics);
  } else if (cmd[0] == "sleep") {
    if (cmd.size() != 2) {
      cout << "Usage: " << cmd[0] << " [number of seconds]" << endl;
      return;
    }
    int nseconds = srsran::string_cast<int>(cmd[1]);
    if (nseconds <= 0) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::seconds(nseconds));
  } else if (cmd[0] == "p") {
    do_padding = !do_padding;
    if (do_padding) {
      cout << "Enter p to stop padding." << endl;
    } else {
      cout << "Enter p to restart padding." << endl;
    }
    control->toggle_padding();
  } else if (cmd[0] == "q") {
    raise(SIGTERM);
  } else if (cmd[0] == "cell_gain") {
    if (cmd.size() != 3) {
      cout << "Usage: " << cmd[0] << " [cell identifier] [gain in dB]" << endl;
      return;
    }

    // Parse command arguments
    uint32_t cell_id = srsran::string_cast<uint32_t>(cmd[1]);
    float    gain_db = srsran::string_cast<float>(cmd[2]);

    // Set cell gain
    control->cmd_cell_gain(cell_id, gain_db);
  } else if (cmd[0] == "flush") {
    if (cmd.size() != 1) {
      cout << "Usage: " << cmd[0] << endl;
      return;
    }
    srslog::flush();
    cout << "Flushed log file buffers" << endl;
  } else {
    cout << "Available commands: " << endl;
    cout << "          t: starts console trace" << endl;
    cout << "          q: quit srsenb" << endl;
    cout << "  cell_gain: set relative cell gain" << endl;
    cout << "      sleep: pauses the commmand line operation for a given time in seconds" << endl;
    cout << "          p: starts MAC padding" << endl;
    cout << "      flush: flushes the buffers for the log file" << endl;
    cout << endl;
  }
}

static void* input_loop(metrics_stdout* metrics, srsenb::enb_command_interface* control)
{
  struct pollfd pfd = {STDIN_FILENO, POLLIN, 0};
  string        input_line;
  while (running) {
    int ret = poll(&pfd, 1, 1000); // query stdin with a timeout of 1000ms
    if (ret == 1) {
      // there is user input to read
      getline(cin, input_line);
      if (cin.eof() || cin.bad()) {
        cout << "Closing stdin thread." << endl;
        break;
      } else if (not input_line.empty()) {
        list<string> cmd_list;
        srsran::string_parse_list(input_line, ';', cmd_list);

        for (const string& cmd : cmd_list) {
          execute_cmd(metrics, control, cmd);
        }
      }
    }
  }
  return nullptr;
}

/// Adjusts the input value in args from kbytes to bytes.
static size_t fixup_log_file_maxsize(int x)
{
  return (x < 0) ? 0 : size_t(x) * 1024u;
}

extern "C" void srsran_dft_exit();
static void     emergency_cleanup_handler(void* data)
{
  srslog::flush();
  if (log_sink) {
    log_sink->flush();
  }
  srsran_dft_exit();
}

static void signal_handler()
{
  running = false;
}

int64_t tstamp_now(void)
{

  struct timespec tms;

  /* The C11 way */
  /* if (! timespec_get(&tms, TIME_UTC))  */

  /* POSIX.1-2008 way */
  if (clock_gettime(CLOCK_REALTIME,&tms)) {
  return -1;
  }
  /* seconds, multiplied with 1 million */
  int64_t micros = tms.tv_sec * 1000000;
  /* Add full microseconds */
  micros += tms.tv_nsec/1000;
  /* round up if necessary */
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return micros;
}

static
enb* enb_instance = nullptr;

template<class T, class Compare>
const T& std_clamp( const T& v, const T& lo, const T& hi, Compare comp )
{
  return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

template<class T>
const T& std_clamp( const T& v, const T& lo, const T& hi )
{
    return std_clamp(v, lo, hi, std::less<T>());
}

static
void fill_mac_stats(mac_ind_data_t* ind)
{
  assert(ind != NULL);
  assert(enb_instance != NULL);
  // print enb_instance
  printf("Main: Xapp Calling: ENB Instance: %p\n", enb_instance);
  ind->msg.tstamp = tstamp_now();

  enb_metrics_t m = {}; 
  bool const rv = enb_instance->get_metrics(&m);

  // print &m
  //printf("Main: Xapp Calling: Metrics struct address: %p\n", (void*)&m);

  //printf("Get metrics returned: %d\n", rv);
  //printf("PHY metrics array size: %zu\n", m.phy.size());
  if (!m.phy.empty()) {
    printf("First PHY metric - DL samples: %d, UL samples: %d\n", 
           m.phy[1].dl.n_samples, m.phy[1].ul.n_samples);
           
  }

  size_t const sz = m.stack.mac.ues.size();
  size_t const rlc_sz = m.stack.pdcp.ues.size();
  if (sz != rlc_sz) {
    fprintf(stderr, "MAC and RLC metrics size mismatch: MAC=%zu, RLC=%zu\n", sz, rlc_sz);
    return;
  }

  ind->msg.len_ue_stats = sz;

  if(sz > 0){
    ind->msg.ue_stats = (mac_ue_stats_impl_t*)calloc(sz, sizeof(mac_ue_stats_impl_t));
    assert(ind->msg.ue_stats != NULL && "Memory exhausted");
  }

  for(size_t i = 0 ; i < sz ; ++i){
    mac_ue_metrics_t const* src = &m.stack.mac.ues[i];
    srsran::rlc_bearer_metrics_t const* rlc_src = &m.stack.rlc.ues[i].bearer[3];

    mac_ue_stats_impl_t* dst = &ind->msg.ue_stats[i];

    //Fahad modifications
  
    // Basic identifiers and metrics
    dst->rnti = src->rnti;
    dst->wb_cqi = src->dl_cqi;        // LTE CQI
    dst->phr = (int8_t)src->phr;      // LTE Power Headroom Report

    // Throughput metrics (4G specific)
    dst->dl_aggr_tbs = (float)src->tx_brate;  // DL throughput
    dst->ul_aggr_tbs = (float)src->rx_brate;  // UL throughput



    // Buffer and packet metrics
    dst->dl_aggr_sdus = src->tx_pkts;     // DL packets
    dst->ul_aggr_sdus = src->rx_pkts;     // UL packets
    dst->bsr = src->ul_buffer;            // UL buffer status

    // Error tracking (using last HARQ index for errors)
    dst->dl_harq[4] = src->tx_errors;     // DL errors
    dst->ul_harq[4] = src->rx_errors;     // UL errors

    // Resource allocation
    dst->dl_aggr_prb = src->allocated_prbs;  // PRB allocation

    // Always use PHY metrics for MCS and SNR in 4G/LTE
    if (i < m.phy.size()) {
      // Read raw SNR values from PHY metrics
      float pusch_sinr = m.phy[i].ul.pusch_sinr;
      float pucch_sinr = m.phy[i].ul.pucch_sinr;

      // Normalize to a display/clamped value. Use -99.9 as sentinel for unavailable or non-finite values
      float displayed_pusch = std::isfinite(pusch_sinr) ? std_clamp<float>(pusch_sinr, -99.9f, 99.9f)
                                                         : -99.9f;
      float displayed_pucch = std::isfinite(pucch_sinr) ? std_clamp<float>(pucch_sinr, -99.9f, 99.9f)
                                                         : -99.9f;

      // Print PHY MCS, SNR, and RSSI values so logs match what is sent to listeners/xApps
      printf("PHY metrics available: UE %zu, DL MCS %f, UL MCS %f, PUSCH_SNR %f, PUCCH_SNR %f, RSSI %f\n",
             i, m.phy[i].dl.mcs, m.phy[i].ul.mcs, displayed_pusch, displayed_pucch, m.phy[i].ul.rssi);

      dst->dl_mcs1 = (uint8_t)std::max(0, (int)std::lround(m.phy[i].dl.mcs));
      dst->ul_mcs1 = (uint8_t)std::max(0, (int)std::lround(m.phy[i].ul.mcs));

      // Store the clamped values in the dst structure (these are what listeners/xApps will see)
      dst->pusch_snr = displayed_pusch;
      dst->pucch_snr = displayed_pucch;
      dst->ul_rssi = m.phy[i].ul.rssi;

    } else {
      dst->dl_mcs1 = 0;
      dst->ul_mcs1 = 0;
      dst->pusch_snr = 0.0f;
      dst->ul_rssi = 0.0f;
    }

  }
}

static inline
uint32_t active_drbs(enb_metrics_t const& m, size_t num_ues)
{
  int nb = 0;
  for(size_t i = 0 ; i <  num_ues ; ++i){
//    for(size_t j =0; j < SRSRAN_N_RADIO_BEARERS; ++j){
//      if( m.stack.rlc.ues[i].bearer[j].num_tx_pdus > 0){ // TODO: get number of active drbs
        nb += 1;
//      }
//    }
  }
  return nb;
}

static
void fill_rlc_stats(rlc_ind_data_t* ind)
{

  assert(ind != NULL);
  assert(enb_instance != NULL);
  ind->msg.tstamp = tstamp_now();

  enb_metrics_t m;;
  bool const rv = enb_instance->get_metrics(&m);

  size_t const num_ues = m.stack.pdcp.ues.size();
  
  uint32_t nb = active_drbs(m, num_ues);
  ind->msg.len = nb;
  if(nb > 0){
    ind->msg.rb = (rlc_radio_bearer_stats_t*)calloc(nb, sizeof(rlc_radio_bearer_stats_t));
    assert(ind->msg.rb!= NULL && "Memory exhausted");
  }

  for(size_t i = 0 ; i <  num_ues ; ++i){
     uint32_t const rnti = m.stack.mac.ues[i].rnti;
    for(size_t j = 0; j < SRSRAN_N_RADIO_BEARERS; ++j){
      //if( m.stack.rlc.ues[i].bearer[j].num_tx_pdus > 0 && nb > 0){ // TODO: check drb is actived
      if (j == 3) { // temp: force to drb id = 3
        srsran::rlc_bearer_metrics_t const* src = &m.stack.rlc.ues[i].bearer[j];
        rlc_radio_bearer_stats_t* dst = &ind->msg.rb[nb - 1];

        dst->txpdu_pkts=src->num_tx_pdus;
        dst->txpdu_bytes=src->num_tx_pdu_bytes;
        dst->rxpdu_pkts=src->num_rx_pdus;
        dst->rxpdu_bytes=src->num_rx_pdu_bytes;

        dst->txsdu_pkts=src->num_tx_sdus;
        dst->txsdu_bytes=(uint32_t)src->num_tx_sdu_bytes;
        dst->rxsdu_pkts=src->num_rx_sdus;
        dst->rxsdu_bytes=(uint32_t)src->num_rx_sdu_bytes;
        dst->rxpdu_dd_pkts=src->num_lost_pdus;
        dst->rxsdu_dd_pkts=src->num_lost_sdus;

        dst->rbid = j;
        dst->rnti = rnti;
        nb -= 1;
      }
    }
  }

}

static
void fill_pdcp_stats(pdcp_ind_data_t* ind)
{
  assert(ind != NULL);
  assert(enb_instance != NULL);
  ind->msg.tstamp = tstamp_now();

  enb_metrics_t m;
  bool const rv = enb_instance->get_metrics(&m);

  size_t const num_ues = m.stack.pdcp.ues.size();
  uint32_t nb = active_drbs(m, num_ues);

  ind->msg.len = nb;
  if(nb > 0){ 
    ind->msg.rb = (pdcp_radio_bearer_stats_t*)calloc(nb, sizeof(pdcp_radio_bearer_stats_t));
    assert(ind->msg.rb!= NULL && "Memory exhausted");
  }

  for(size_t i = 0 ; i < num_ues ; ++i){
    uint32_t const rnti = m.stack.mac.ues[i].rnti;
    for(size_t j =0; j < SRSRAN_N_RADIO_BEARERS; ++j){
      //if(m.stack.pdcp.ues[i].bearer[j].num_tx_pdus > 0 && nb > 0){ // TODO: check drb is actived
      if (j == 3) { // temp: force to drb id = 3
        srsran::pdcp_bearer_metrics_t const* src = &m.stack.pdcp.ues[i].bearer[j];
        pdcp_radio_bearer_stats_t* dst = &ind->msg.rb[nb - 1];

        dst->txpdu_pkts=src->num_tx_pdus; 
        dst->txpdu_bytes=src->num_tx_pdu_bytes;
        dst->rxpdu_pkts=src->num_rx_pdus; 
        dst->rxpdu_bytes=src->num_rx_pdu_bytes;
        dst->rbid = j;
        dst->rnti = rnti;
        nb -= 1;
      }
    }
  }

}

static
void read_slice_conf(slice_conf_t* conf) {
  srsran_assert(conf != NULL, "conf == NULL");

  /// read_slice_conf DL ///
  Slicing &slice_stats = Slicing::getInstance();
  ul_dl_slice_conf_t* stats_dl = &slice_stats.stats_slice_conf.dl;
  ul_dl_slice_conf_t* rd_dl = &conf->dl;

  // get sched algo
  rd_dl->len_sched_name = stats_dl->len_sched_name;
  rd_dl->sched_name = (char*)malloc(rd_dl->len_sched_name);
  srsran_assert(rd_dl->sched_name != NULL, "memory exhausted");
  memcpy(rd_dl->sched_name, stats_dl->sched_name, rd_dl->len_sched_name);


  // get num of slice
  rd_dl->len_slices = stats_dl->len_slices;

  if (rd_dl->len_slices > 0) {
    // get slice algo
    slice_algorithm_e algo = stats_dl->slices[0].params.type;

    // allocate memory to read multi slices
    rd_dl->slices = (fr_slice_t*)calloc(rd_dl->len_slices, sizeof(fr_slice_t));
    srsran_assert(rd_dl->slices != NULL, "memory exhausted");

    // Get each slice config: id, label, sched algo, slice algo data
    for (uint32_t i = 0; i < rd_dl->len_slices; ++i) {
      fr_slice_t* rd_slice = &rd_dl->slices[i];
      fr_slice_t* st_slice = &stats_dl->slices[i];

      // get id
      rd_dl->slices[i].id = st_slice->id;

      // get label
      rd_slice->len_label = st_slice->len_label;
      rd_slice->label     = (char*)malloc(rd_slice->len_label);
      srsran_assert(rd_slice->label != NULL, "memory exhausted");
      memcpy(rd_slice->label, st_slice->label, rd_slice->len_label);

      // get sched algo
      rd_slice->len_sched = st_slice->len_sched;
      rd_slice->sched     = (char*)malloc(rd_slice->len_sched);
      srsran_assert(rd_slice->sched != NULL, "memory exhausted");
      memcpy(rd_slice->sched, st_slice->sched, rd_slice->len_sched);

      // get slice algo data
      if (algo == SLICE_ALG_SM_V0_STATIC) {
        rd_slice->params.type = st_slice->params.type;
        static_slice_t* sta   = &rd_slice->params.u.sta;
        sta->pos_high         = st_slice->params.u.sta.pos_high;
        sta->pos_low          = st_slice->params.u.sta.pos_low;
      } else {
        srsran_assert(0 != 0, "Unknow type of DL algo\n");
      }
    }
  }

  /// TODO: read_slice_conf UL ///
  ul_dl_slice_conf_t* rd_ul = &conf->ul;
  char const* ul_algo_name = "round_round_ul";
  rd_ul->len_sched_name = strlen(ul_algo_name);
  rd_ul->sched_name = (char*)malloc(rd_ul->len_sched_name);
  srsran_assert(rd_ul->sched_name != NULL, "memory exhausted");
  memcpy(rd_ul->sched_name, ul_algo_name, rd_ul->len_sched_name);

}

static
void read_ue_slice_conf(ue_slice_conf_t* rd_ue) {
  srsran_assert(rd_ue != NULL, "conf == NULL");

  enb_metrics_t m;
  bool const rv = enb_instance->get_metrics(&m);

  /// read_ue_slice_conf DL ///
  Slicing &slice_stats = Slicing::getInstance();
  ue_slice_conf_t* stats_ue_s = &slice_stats.stats_ue_slice_conf;
  rd_ue->len_ue_slice = stats_ue_s->len_ue_slice;

  if(rd_ue->len_ue_slice > 0){
    rd_ue->ues = (ue_slice_assoc_t*)calloc(stats_ue_s->len_ue_slice, sizeof(ue_slice_assoc_t));
    srsran_assert(rd_ue->ues != NULL, "Memory exhausted");
  }

  for (uint32_t i = 0; i < rd_ue->len_ue_slice; i++) {
    rd_ue->ues[i].rnti = stats_ue_s->ues[i].rnti;
    rd_ue->ues[i].dl_id = stats_ue_s->ues[i].dl_id;
  }
  /// TODO: read_ue_slice_conf UL ///
}

static
void fill_slice_stats(slice_ind_data_t* ind)
{
  srsran_assert(ind != NULL, "ind == NULL");
  ind->msg.tstamp = tstamp_now();

  read_slice_conf(&ind->msg.slice_conf);
  read_ue_slice_conf(&ind->msg.ue_slice_conf);
}

static
void read_RAN(sm_ag_if_rd_t* data)
{
  assert(data != NULL);

  if(data->type == MAC_STATS_V0){
    fill_mac_stats(&data->mac_stats);
  } else if (data->type == RLC_STATS_V0 ){
    fill_rlc_stats(&data->rlc_stats); 
  } else if(data->type == PDCP_STATS_V0){
    fill_pdcp_stats(&data->pdcp_stats);
  } else if(data->type == SLICE_STATS_V0){
    fill_slice_stats(&data->slice_stats);
  } else {
    assert(0!=0 && "Unknown data type");
  }

}

static
sm_ag_if_ans_t write_slice(slice_ctrl_req_data_t const& s)
{
  assert(enb_instance != NULL);

  enb_stack_base& stack_base =  enb_instance->get_eutra_stack();
  sm_ag_if_ans_t ans = {};
  ans.type = SLICE_AGENT_IF_CTRL_ANS_V0;
  try{
    enb_stack_lte& stack = dynamic_cast<enb_stack_lte&>(stack_base);
    ans.slice.ans = stack.slice(s);
    if (ans.slice.ans != 0)
      printf("ans.slice.ans == SLICE_CTRL_OUT_ERROR\n");
  } catch(std::bad_cast const& e){
    std::cout << "Exception thrown while casting\n";
    exit(-1);
  } catch (...){
    std::cout << "Unknown exception thrown\n";
    exit(-1);
  }

  return ans;
}



static
sm_ag_if_ans_t write_RAN(sm_ag_if_wr_t const* data)
{
  assert(data != NULL);
  assert(enb_instance != NULL);

  sm_ag_if_ans_t ans = {};
  ans.type = SM_AGENT_IF_ANS_V0_END;
  if(data->type == SLICE_CTRL_REQ_V0 ){
   ans = write_slice(data->slice_req_ctrl); 
  } else {
    assert(0!=0 && "unknown data type");
  }
  assert(ans.type != SM_AGENT_IF_ANS_V0_END);
  puts("Write RAN function called \n");
 
  return ans;
}

int main(int argc, char* argv[])
{
  srsran_register_signal_handler(signal_handler);
  add_emergency_cleanup_handler(emergency_cleanup_handler, nullptr);

  all_args_t                         args = {};
  srsran::metrics_hub<enb_metrics_t> metricshub;
  metrics_stdout                     metrics_screen;

  cout << "---  Software Radio Systems LTE eNodeB  ---" << endl << endl;

  srsran_debug_handle_crash(argc, argv);
  parse_args(&args, argc, argv);

  // Setup the default log sink.
  srslog::set_default_sink(
      (args.log.filename == "stdout")
          ? srslog::fetch_stdout_sink()
          : srslog::fetch_file_sink(args.log.filename, fixup_log_file_maxsize(args.log.file_max_size)));

  // Alarms log channel creation.
  srslog::sink&        alarm_sink     = srslog::fetch_file_sink(args.general.alarms_filename, 0, true);
  srslog::log_channel& alarms_channel = srslog::fetch_log_channel("alarms", alarm_sink, {"ALRM", '\0', false});
  alarms_channel.set_enabled(args.general.alarms_log_enable);

#ifdef ENABLE_SRSLOG_EVENT_TRACE
  if (args.general.tracing_enable) {
    if (!srslog::event_trace_init(args.general.tracing_filename, args.general.tracing_buffcapacity)) {
      return SRSRAN_ERROR;
    }
  }
#endif

  // Start the log backend.
  srslog::init();

  srslog::fetch_basic_logger("ALL").set_level(srslog::basic_levels::warning);
  srslog::fetch_basic_logger("POOL").set_level(srslog::basic_levels::warning);
  srsran::log_args(argc, argv, "ENB");

  srsran::check_scaling_governor(args.rf.device_name);

  // Set up the JSON log channel used by metrics and events.
  srslog::sink& json_sink =
      srslog::fetch_file_sink(args.general.report_json_filename, 0, false, srslog::create_json_formatter());
  srslog::log_channel& json_channel = srslog::fetch_log_channel("JSON_channel", json_sink, {});
  json_channel.set_enabled(args.general.report_json_enable);

  // Configure the event logger just before starting the eNB class.
  if (args.general.report_json_enable) {
    event_logger::asn1_output_format format = (args.general.report_json_asn1_oct)
                                                  ? event_logger::asn1_output_format::octets
                                                  : event_logger::asn1_output_format::text;
    event_logger::configure(json_channel, format);
  }

  if (mlockall((uint32_t)MCL_CURRENT | (uint32_t)MCL_FUTURE) == -1) {
    srsran::console("Failed to `mlockall`: {}", errno);
  }

  // Create eNB
  unique_ptr<srsenb::enb> enb{new srsenb::enb(srslog::get_default_sink())};
  if (enb->init(args) != SRSRAN_SUCCESS) {
    enb->stop();
    return SRSRAN_ERROR;
  }
  //  printf enb.get()
  cout << "---  eNodeB initialized  ---" << endl;
  cout << "eNB Instance Address: " << enb.get() << endl;

  // Set metrics
  metricshub.init(enb.get(), args.general.metrics_period_secs);
  metricshub.add_listener(&metrics_screen);
  metrics_screen.set_handle(enb.get());

  // ADDED IN
  std::string metrics_load = "/tmp/enb_metrics_" + std::to_string(args.enb.n_prb) + ".csv";
  std::cout << "\nMetrics File" << metrics_load << std::endl;
  srsenb::metrics_csv metrics_file(metrics_load);

  // srsenb::metrics_csv metrics_file(args.general.metrics_csv_filename);
  if (args.general.metrics_csv_enable) {
    metricshub.add_listener(&metrics_file);
    metrics_file.set_handle(enb.get());
  }

  srsenb::metrics_json json_metrics(json_channel, enb.get());
  if (args.general.report_json_enable) {
    metricshub.add_listener(&json_metrics);
  }

  // create input thread
  std::thread input(&input_loop, &metrics_screen, (enb_command_interface*)enb.get());


//////////////////////////////////
//// Init the E2 Agent
//////////////////////////////////

// fix this
  // static enb* enb_instance = nullptr;

  enb_instance = enb.get();
  // print enb_instance
  //printf("ENB Instance Address: %p\n", enb_instance);
  assert(enb_instance != NULL);

  std::string mcc_str, mnc_str;
  srsran::mcc_to_string(args.stack.s1ap.mcc, &mcc_str);
  srsran::mnc_to_string(args.stack.s1ap.mnc, &mnc_str);
  uint16_t const mcc = atoi(mcc_str.c_str());
  uint16_t const mnc = atoi(mnc_str.c_str());
  int const enb_id = args.enb.enb_id;

  const int mnc_digit_len = 2; 
  sm_io_ag_t io = {.read = read_RAN, .write = write_RAN};

  cout << "[E2 NODE]: mcc = " << mcc
       << " mnc = " << mnc
       << " mnc_digit = " << mnc_digit_len
       << " enb_id = " << enb_id << endl;

  int const agent_argc = 1;
  fr_args_t ric_args = {{0}};
  char* conf_dir = getenv("FLEXRIC_CONF");
  char* lib_dir = getenv("FLEXRIC_LIB_DIR");

  if (conf_dir != NULL)
    strcpy(ric_args.conf_file, conf_dir);
  else
    strcpy(ric_args.conf_file, "/usr/local/etc/flexric/flexric.conf");
  if (lib_dir != NULL)
    strcpy(ric_args.libs_dir, lib_dir);
  else
    strcpy(ric_args.libs_dir, "/usr/local/lib/flexric/");
  init_agent_api(mcc, mnc, mnc_digit_len, enb_id,0, ngran_gNB, io, &ric_args);

//////////////////////////////////
/////////////////////////////////


  if (running) {
    if (args.gui.enable) {
      enb->start_plot();
    }
  }
  int cnt    = 0;
  int ts_cnt = 0;
  while (running) {
    if (args.general.print_buffer_state) {
      cnt++;
      if (cnt == 1000) {
        cnt = 0;
        enb->print_pool();
      }
    }
    if (stdout_ts_enable) {
      if (++ts_cnt == 100) {
        ts_cnt = 0;
        char        buff[64];
        std::time_t t = std::time(nullptr);
        if (std::strftime(buff, sizeof(buff), "%FT%T", std::gmtime(&t))) {
          std::cout << buff << '\n';
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  input.join();
  metricshub.stop();
  enb->stop();
  cout << "---  exiting  ---" << endl;

  return SRSRAN_SUCCESS;
}
