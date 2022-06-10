// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington, Ahmed ElTantawy,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpgpu_sim_wrapper.h"
#include <sys/stat.h>
#define SP_BASE_POWER 0
#define SFU_BASE_POWER 0

static const char* pwr_cmp_label[] = {
    "IBP,", "ICP,",  "DCP,",   "TCP,",   "CCP,",        "SHRDP,",
    "RFP,", "SPP,",  "SFUP,",  "FPUP,",  "SCHEDP,",     "L2CP,",
    "MCP,", "NOCP,", "DRAMP,", "PIPEP,", "IDLE_COREP,", "CONST_DYNAMICP"};

enum pwr_cmp_t {
  IBP = 0,
  ICP,
  DCP,
  TCP,
  CCP,
  SHRDP,
  RFP,
  SPP,
  SFUP,
  FPUP,
  SCHEDP,
  L2CP,
  MCP,
  NOCP,
  DRAMP,
  PIPEP,
  IDLE_COREP,
  CONST_DYNAMICP,
  NUM_COMPONENTS_MODELLED
};

gpgpu_sim_wrapper::gpgpu_sim_wrapper(bool power_simulation_enabled,
                                     char* xmlfile,unsigned number_shader,double* m_cluster_freq) {
  kernel_sample_count = 0;
  total_sample_count = 0;

  kernel_tot_power = 0;

  num_pwr_cmps = NUM_COMPONENTS_MODELLED;
  num_perf_counters = NUM_PERFORMANCE_COUNTERS;
  number_shaders = number_shader;
 // printf("\nnum_pwr_cmps in wrapper.cc:%d",NUM_COMPONENTS_MODELLED);
  // Initialize per-component counter/power vectors
  avg_max_min_counters<double> init;
  kernel_cmp_pwr.resize(NUM_COMPONENTS_MODELLED, init);
  kernel_cmp_perf_counters.resize(NUM_PERFORMANCE_COUNTERS, init);

  kernel_power = init;   // Per-kernel powers
  gpu_tot_power = init;  // Global powers

  sample_cmp_pwr.resize(NUM_COMPONENTS_MODELLED, 0);
  Throughput.resize(number_shader);
  sample_cmp_pwr_per_core.resize(number_shader);
  sample_cmp_pwr_S.resize(number_shader);
  for(int i=0;i<number_shader;i++)
    sample_cmp_pwr_per_core[i].resize(NUM_COMPONENTS_MODELLED,0);
  //printf("\nnumber of shader %d",number_shader);
  sample_perf_counters.resize(NUM_PERFORMANCE_COUNTERS, 0);
  initpower_coeff.resize(NUM_PERFORMANCE_COUNTERS, 0);
  effpower_coeff.resize(NUM_PERFORMANCE_COUNTERS, 0);

  const_dynamic_power = 0;
  proc_power = 0;
  g_power_filename = NULL;
  g_power_trace_filename = NULL;
  g_metric_trace_filename = NULL;
  g_steady_state_tracking_filename = NULL;
  xml_filename = xmlfile;
  g_power_simulation_enabled = power_simulation_enabled;
  g_power_trace_enabled = false;
  g_steady_power_levels_enabled = false;
  g_power_trace_zlevel = 0;
  g_power_per_cycle_dump = false;
  gpu_steady_power_deviation = 0;
  gpu_steady_min_period = 0;
  n_cluster_freq = m_cluster_freq;
  gpu_stat_sample_freq = 0;
  p = new ParseXML();
  p_cores = (ParseXML**)malloc(sizeof(ParseXML*)*number_shader);
  power_per_core = (double*)malloc(sizeof(double)*number_shader);
  if (g_power_simulation_enabled) {
    p->parse(xml_filename);
//    p->sys.core[0].clock_rate = (int)(cluster_freq[0]/1000000);
    for(int i=0;i<number_shader;i++) {
      p_cores[i] = new ParseXML();
      p_cores[i]->parse(xml_filename);
      p_cores[i]->sys.core[0].clock_rate = (int)(n_cluster_freq[i]/((1e6)));

    }
  }
  proc = new Processor(p);
  proc_cores = new Processor*[number_shader];

  for(int i=0;i<number_shader;i++)
  {
    proc_cores[i] = new Processor(p_cores[i]);

  }
  power_trace_file = NULL;
  metric_trace_file = NULL;
  steady_state_tacking_file = NULL;
  has_written_avg = false;
  init_inst_val = false;
}

gpgpu_sim_wrapper::~gpgpu_sim_wrapper() {}

bool gpgpu_sim_wrapper::sanity_check(double a, double b) {
  if (b == 0)
    return (abs(a - b) < 0.00001);
  else
    return (abs(a - b) / abs(b) < 0.00001);

  return false;
}
void gpgpu_sim_wrapper::init_mcpat(
    char* xmlfile, char* powerfilename, char* power_trace_filename,
    char* metric_trace_filename, char* steady_state_filename,
    bool power_sim_enabled, bool trace_enabled, bool steady_state_enabled,
    bool power_per_cycle_dump, double steady_power_deviation,
    double steady_min_period, int zlevel, double init_val,
    int stat_sample_freq) {
  // Write File Headers for (-metrics trace, -power trace)

  reset_counters();
  static bool mcpat_init = true;

  // initialize file name if it is not set
  time_t curr_time;
  time(&curr_time);
  char* date = ctime(&curr_time);
  char* s = date;
  while (*s) {
    if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
    if (*s == '\n' || *s == '\r') *s = 0;
    s++;
  }

  if (mcpat_init) {
    g_power_filename = powerfilename;
    g_power_trace_filename = power_trace_filename;
    g_metric_trace_filename = metric_trace_filename;
    g_steady_state_tracking_filename = steady_state_filename;
    xml_filename = xmlfile;
    g_power_simulation_enabled = power_sim_enabled;
    g_power_trace_enabled = trace_enabled;
    g_steady_power_levels_enabled = steady_state_enabled;
    g_power_trace_zlevel = zlevel;
    g_power_per_cycle_dump = power_per_cycle_dump;
    gpu_steady_power_deviation = steady_power_deviation;
    gpu_steady_min_period = steady_min_period;

    gpu_stat_sample_freq = stat_sample_freq;

    // p->sys.total_cycles=gpu_stat_sample_freq*4;
    p->sys.total_cycles = gpu_stat_sample_freq;
    for(int i=0;i<number_shaders;i++)
      p_cores[i]->sys.total_cycles = gpu_stat_sample_freq*n_cluster_freq[i]/n_cluster_freq[0];


    power_trace_file = NULL;
    metric_trace_file = NULL;
    steady_state_tacking_file = NULL;

    if (g_power_trace_enabled) {
      power_trace_file = gzopen(g_power_trace_filename, "w");
      metric_trace_file = gzopen(g_metric_trace_filename, "w");
      if ((power_trace_file == NULL) || (metric_trace_file == NULL)) {
        printf("error - could not open trace files \n");
        exit(1);
      }
      gzsetparams(power_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);

      gzprintf(power_trace_file, "power,");
      for (unsigned i = 0; i < num_pwr_cmps; i++) {
        gzprintf(power_trace_file, pwr_cmp_label[i]);
      }
      gzprintf(power_trace_file, "\n");

      gzsetparams(metric_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);
      for (unsigned i = 0; i < num_perf_counters; i++) {
        gzprintf(metric_trace_file, perf_count_label[i]);
      }
      gzprintf(metric_trace_file, "\n");

      gzclose(power_trace_file);
      gzclose(metric_trace_file);
    }
    if (g_steady_power_levels_enabled) {
      steady_state_tacking_file = gzopen(g_steady_state_tracking_filename, "w");
      if ((steady_state_tacking_file == NULL)) {
        printf("error - could not open trace files \n");
        exit(1);
      }
      gzsetparams(steady_state_tacking_file, g_power_trace_zlevel,
                  Z_DEFAULT_STRATEGY);
      gzprintf(steady_state_tacking_file, "start,end,power,IPC,");
      for (unsigned i = 0; i < num_perf_counters; i++) {
        gzprintf(steady_state_tacking_file, perf_count_label[i]);
      }
      gzprintf(steady_state_tacking_file, "\n");

      gzclose(steady_state_tacking_file);
    }

    mcpat_init = false;
    has_written_avg = false;
    powerfile.open(g_power_filename);
    int flg = chmod(g_power_filename, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    assert(flg == 0);
  }
  sample_val = 0;
  init_inst_val = init_val;  // gpu_tot_sim_insn+gpu_sim_insn;
}

void gpgpu_sim_wrapper::reset_counters() {
  avg_max_min_counters<double> init;
  for (unsigned i = 0; i < num_perf_counters; ++i) {
    sample_perf_counters[i] = 0;
    kernel_cmp_perf_counters[i] = init;
  }
  for (unsigned i = 0; i < num_pwr_cmps; ++i) {
    sample_cmp_pwr[i] = 0;
    kernel_cmp_pwr[i] = init;
    for (int j=0;j<number_shaders;j++)
      sample_cmp_pwr_per_core[j][i] = 0;
  }

  // Reset per-kernel counters
  kernel_sample_count = 0;
  kernel_tot_power = 0;
  kernel_power = init;

  return;
}
ParseXML* gpgpu_sim_wrapper::return_p() {
  return p;
}


void gpgpu_sim_wrapper::set_inst_power(bool clk_gated_lanes, double tot_cycles,
                                       double busy_cycles, double tot_inst,
                                       double int_inst, double fp_inst,
                                       double load_inst, double store_inst,
                                       double committed_inst,
                                       double *tot_ins_set_inst_power,double *total_int_ins_set_inst_power,
                                       double *tot_fp_ins_set_inst_power,double *tot_commited_ins_set_inst_power,double* cluster_freq, unsigned stat_sample_freq) {

  p->sys.core[0].clock_rate =  (int)(cluster_freq[0]/((1e6)));
  p->sys.core[0].gpgpu_clock_gated_lanes = clk_gated_lanes;
  p->sys.core[0].total_cycles = tot_cycles;
  p->sys.core[0].busy_cycles = busy_cycles;

FILE * file;
file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
fprintf(file,"\n********set_inst_power**********");
fprintf(file,"\nclk_gated_lanes: %i",clk_gated_lanes);
fprintf(file,"\ntot_cycles: %10.7lf",tot_cycles);
fprintf(file,"\nbusy_cycles: %10.7lf",busy_cycles);
fprintf(file,"\nint_inst: %10.7lf",int_inst);
fprintf(file,"\nfp_inst: %10.7lf",fp_inst);
fprintf(file,"\nload_inst: %10.7lf",load_inst);
fprintf(file,"\nstore_inst: %10.7lf",store_inst);
fprintf(file,"\ncommitted_inst: %10.7lf",committed_inst);
fclose(file);

  for(int i=0;i<number_shaders;i++)
  {
    Throughput[i] = tot_ins_set_inst_power[i]/ (cluster_freq[i]/cluster_freq[0]);
     printf("\nclock rate wrapper is : %d",p_cores[i]->sys.core[0].clock_rate);
    p_cores[i]->sys.core[0].clock_rate =  (int)(cluster_freq[i]/((1e6)));
    p_cores[i]->sys.total_cycles = tot_cycles*cluster_freq[i]/cluster_freq[0];
    p_cores[i]->sys.core[0].gpgpu_clock_gated_lanes = clk_gated_lanes;
    p_cores[i]->sys.core[0].total_cycles = tot_cycles*cluster_freq[i]/cluster_freq[0];
    p_cores[i]->sys.core[0].busy_cycles = busy_cycles*cluster_freq[i]/cluster_freq[0];
    p_cores[i]->sys.core[0].total_instructions =
        tot_ins_set_inst_power[i] * p_cores[i]->sys.scaling_coefficients[TOT_INST];
    p_cores[i]->sys.core[0].int_instructions =
        total_int_ins_set_inst_power[i] * p_cores[i]->sys.scaling_coefficients[FP_INT];
    p_cores[i]->sys.core[0].fp_instructions =
        tot_fp_ins_set_inst_power[i] * p_cores[i]->sys.scaling_coefficients[FP_INT];
    p_cores[i]->sys.core[0].load_instructions = 0;
    p_cores[i]->sys.core[0].store_instructions = 0;
    p_cores[i]->sys.core[0].committed_instructions = tot_commited_ins_set_inst_power[i];
//    fprintf(file,"%lf ",p_cores[i]->sys.core[0].total_cycles);
  }

  p->sys.core[0].total_instructions =
      tot_inst * p->sys.scaling_coefficients[TOT_INST];
  p->sys.core[0].int_instructions =
      int_inst * p->sys.scaling_coefficients[FP_INT];
  p->sys.core[0].fp_instructions =
      fp_inst * p->sys.scaling_coefficients[FP_INT];
  p->sys.core[0].load_instructions = load_inst;
  p->sys.core[0].store_instructions = store_inst;
  p->sys.core[0].committed_instructions = committed_inst;
  sample_perf_counters[FP_INT] = int_inst + fp_inst;
  sample_perf_counters[TOT_INST] = tot_inst;
}

void gpgpu_sim_wrapper::set_regfile_power(double reads, double writes,
                                          double ops,double *reads_regfile_power,double *write_regfile_power,double *ops_regfile_power) {
  p->sys.core[0].int_regfile_reads =
      reads * p->sys.scaling_coefficients[REG_RD];
  p->sys.core[0].int_regfile_writes =
      writes * p->sys.scaling_coefficients[REG_WR];
  p->sys.core[0].non_rf_operands =
      ops * p->sys.scaling_coefficients[NON_REG_OPs];

 // printf("\n p->sys.core[0].clock_rate: %d", p->sys.core[0].clock_rate);

  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_regfile_power**********");
  fprintf(file,"\nreads: %10.7lf",reads);
  fprintf(file,"\nwrites: %10.7lf",writes);
  fprintf(file,"\nops: %10.7lf",ops);
  fclose(file);

  //  Per core data
  for(int i=0;i<number_shaders;i++)
  {
   // printf("\n p_cores[i]->sys.core[0].clock_rate: %d", p_cores[i]->sys.core[0].clock_rate);
    p_cores[i]->sys.core[0].int_regfile_reads =
        reads_regfile_power[i] * p_cores[i]->sys.scaling_coefficients[REG_RD];
    p_cores[i]->sys.core[0].int_regfile_writes =
        write_regfile_power[i] * p_cores[i]->sys.scaling_coefficients[REG_WR];
    p_cores[i]->sys.core[0].non_rf_operands =
        ops_regfile_power[i] * p_cores[i]->sys.scaling_coefficients[NON_REG_OPs];
  }

    printf("\nset_regfile_power");
  for(int i=0;i<number_shaders;i++){
    printf("\n%lf %lf %lf %lf %lf %lf  ", reads_regfile_power[i],write_regfile_power[i],ops_regfile_power[i],p_cores[i]->sys.core[0].int_regfile_reads,\
           p_cores[i]->sys.core[0].int_regfile_writes,p_cores[i]->sys.core[0].non_rf_operands);
  }
  printf("\n%lf %lf %lf %lf %lf %lf ",reads,writes,ops,p->sys.core[0].int_regfile_reads,p->sys.core[0].int_regfile_writes,p->sys.core[0].non_rf_operands);


  sample_perf_counters[REG_RD] = reads;
  sample_perf_counters[REG_WR] = writes;
  sample_perf_counters[NON_REG_OPs] = ops;
}

void gpgpu_sim_wrapper::set_icache_power(double hits, double misses) {
  p->sys.core[0].icache.read_accesses =
      hits * p->sys.scaling_coefficients[IC_H] +
      misses * p->sys.scaling_coefficients[IC_M];
  p->sys.core[0].icache.read_misses =
      misses * p->sys.scaling_coefficients[IC_M];

  for(int i=0;i<number_shaders;i++){
    p_cores[i]->sys.core[0].icache.read_accesses =
        0 * p_cores[i]->sys.scaling_coefficients[IC_H] +
        0 * p_cores[i]->sys.scaling_coefficients[IC_M];
    p_cores[i]->sys.core[0].icache.read_misses =
        0 * p_cores[i]->sys.scaling_coefficients[IC_M];
  }
  sample_perf_counters[IC_H] = hits;
  sample_perf_counters[IC_M] = misses;
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_icache_power**********");
  fprintf(file,"\nhits: %10.7lf",hits);
  fprintf(file,"\nmisses: %10.7lf",misses);
  fclose(file);
}

void gpgpu_sim_wrapper::set_ccache_power(double hits, double misses) {
  p->sys.core[0].ccache.read_accesses =
      hits * p->sys.scaling_coefficients[CC_H] +
      misses * p->sys.scaling_coefficients[CC_M];
  p->sys.core[0].ccache.read_misses =
      misses * p->sys.scaling_coefficients[CC_M];

  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.core[0].ccache.read_accesses =
        0 *p_cores[i]->sys.scaling_coefficients[CC_H] +
        0 *  p_cores[i]->sys.scaling_coefficients[CC_M];
    p_cores[i]->sys.core[0].ccache.read_misses =
        0 * p_cores[i]->sys.scaling_coefficients[CC_M];
  }
  sample_perf_counters[CC_H] = hits;
  sample_perf_counters[CC_M] = misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_ccache_power**********");
  fprintf(file,"\nhits: %10.7lf",hits);
  fprintf(file,"\nmisses: %10.7lf",misses);
  fclose(file);
}

void gpgpu_sim_wrapper::set_tcache_power(double hits, double misses) {
  p->sys.core[0].tcache.read_accesses =
      hits * p->sys.scaling_coefficients[TC_H] +
      misses * p->sys.scaling_coefficients[TC_M];
  p->sys.core[0].tcache.read_misses =
      misses * p->sys.scaling_coefficients[TC_M];

  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.core[0].tcache.read_accesses =
        0 * p_cores[i]->sys.scaling_coefficients[TC_H] +
        0 * p_cores[i]->sys.scaling_coefficients[TC_M];
    p_cores[i]->sys.core[0].tcache.read_misses =
        0 * p_cores[i]->sys.scaling_coefficients[TC_M];
  }

  sample_perf_counters[TC_H] = hits;
  sample_perf_counters[TC_M] = misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_tcache_power**********");
  fprintf(file,"\nhits: %10.7lf",hits);
  fprintf(file,"\nmisses: %10.7lf",misses);
  fclose(file);
}

void gpgpu_sim_wrapper::set_shrd_mem_power(double accesses,double *shmem_read_set_power) {
  p->sys.core[0].sharedmemory.read_accesses =
      accesses * p->sys.scaling_coefficients[SHRD_ACC];
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_shrd_mem_power**********");
  fprintf(file,"\naccesses: %10.7lf",accesses);
  fclose(file);
  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.core[0].sharedmemory.read_accesses =
        shmem_read_set_power[i] * p_cores[i]->sys.scaling_coefficients[SHRD_ACC];
  }

  printf("\nset_shrd_mem_power");
  for(int i=0;i<number_shaders;i++)
    printf("\n%lf",p_cores[i]->sys.core[0].sharedmemory.read_accesses);
  printf("\n%lf",p->sys.core[0].sharedmemory.read_accesses);

  sample_perf_counters[SHRD_ACC] = accesses;
}

void gpgpu_sim_wrapper::set_l1cache_power(double read_hits, double read_misses,
                                          double write_hits,
                                          double write_misses) {

  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_l1cache_power**********");
  fprintf(file,"\nread_hits: %10.7lf",read_hits);
  fprintf(file,"\nread_misses: %10.7lf",read_misses);
  fprintf(file,"\nwrite_hits: %10.7lf",write_hits);
  fprintf(file,"\nwrite_misses: %10.7lf",write_misses);
  fclose(file);

  p->sys.core[0].dcache.read_accesses =
      read_hits * p->sys.scaling_coefficients[DC_RH] +
      read_misses * p->sys.scaling_coefficients[DC_RM];
  p->sys.core[0].dcache.read_misses =
      read_misses * p->sys.scaling_coefficients[DC_RM];
  p->sys.core[0].dcache.write_accesses =
      write_hits * p->sys.scaling_coefficients[DC_WH] +
      write_misses * p->sys.scaling_coefficients[DC_WM];
  p->sys.core[0].dcache.write_misses =
      write_misses * p->sys.scaling_coefficients[DC_WM];

  //  Per core data
  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.core[0].dcache.read_accesses =
        0 * p_cores[i]->sys.scaling_coefficients[DC_RH] +
        0 * p_cores[i]->sys.scaling_coefficients[DC_RM];
    p_cores[i]->sys.core[0].dcache.read_misses =
        0 * p_cores[i]->sys.scaling_coefficients[DC_RM];
    p_cores[i]->sys.core[0].dcache.write_accesses =
        0 * p_cores[i]->sys.scaling_coefficients[DC_WH] +
        0 * p_cores[i]->sys.scaling_coefficients[DC_WM];
    p_cores[i]->sys.core[0].dcache.write_misses =
        0 * p_cores[i]->sys.scaling_coefficients[DC_WM];
  }
  sample_perf_counters[DC_RH] = read_hits;
  sample_perf_counters[DC_RM] = read_misses;
  sample_perf_counters[DC_WH] = write_hits;
  sample_perf_counters[DC_WM] = write_misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
}

void gpgpu_sim_wrapper::set_l2cache_power(double read_hits, double read_misses,
                                          double write_hits,
                                          double write_misses) {
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_l2cache_power**********");
  fprintf(file,"\nread_hits: %10.7lf",read_hits);
  fprintf(file,"\nread_misses: %10.7lf",read_misses);
  fprintf(file,"\nwrite_hits: %10.7lf",write_hits);
  fprintf(file,"\nwrite_misses: %10.7lf",write_misses);
  fclose(file);

  p->sys.l2.total_accesses = read_hits * p->sys.scaling_coefficients[L2_RH] +
                             read_misses * p->sys.scaling_coefficients[L2_RM] +
                             write_hits * p->sys.scaling_coefficients[L2_WH] +
                             write_misses * p->sys.scaling_coefficients[L2_WM];
  p->sys.l2.read_accesses = read_hits * p->sys.scaling_coefficients[L2_RH] +
                            read_misses * p->sys.scaling_coefficients[L2_RM];
  p->sys.l2.write_accesses = write_hits * p->sys.scaling_coefficients[L2_WH] +
                             write_misses * p->sys.scaling_coefficients[L2_WM];
  p->sys.l2.read_hits = read_hits * p->sys.scaling_coefficients[L2_RH];
  p->sys.l2.read_misses = read_misses * p->sys.scaling_coefficients[L2_RM];
  p->sys.l2.write_hits = write_hits * p->sys.scaling_coefficients[L2_WH];
  p->sys.l2.write_misses = write_misses * p->sys.scaling_coefficients[L2_WM];

  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.l2.total_accesses = 0 * p_cores[i]->sys.scaling_coefficients[L2_RH] +
                                        0 * p_cores[i]->sys.scaling_coefficients[L2_RM] +
                                        0 * p_cores[i]->sys.scaling_coefficients[L2_WH] +
                                        0 * p_cores[i]->sys.scaling_coefficients[L2_WM];
    p_cores[i]->sys.l2.read_accesses = 0 * p_cores[i]->sys.scaling_coefficients[L2_RH] +
                                       0 * p_cores[i]->sys.scaling_coefficients[L2_RM];
    p_cores[i]->sys.l2.write_accesses = 0 * p_cores[i]->sys.scaling_coefficients[L2_WH] +
                                        0 * p_cores[i]->sys.scaling_coefficients[L2_WM];
    p_cores[i]->sys.l2.read_hits = 0 * p_cores[i]->sys.scaling_coefficients[L2_RH];
    p_cores[i]->sys.l2.read_misses = 0 * p_cores[i]->sys.scaling_coefficients[L2_RM];
    p_cores[i]->sys.l2.write_hits = 0 * p_cores[i]->sys.scaling_coefficients[L2_WH];
    p_cores[i]->sys.l2.write_misses = 0 * p_cores[i]->sys.scaling_coefficients[L2_WM];
  }

  sample_perf_counters[L2_RH] = read_hits;
  sample_perf_counters[L2_RM] = read_misses;
  sample_perf_counters[L2_WH] = write_hits;
  sample_perf_counters[L2_WM] = write_misses;
}

void gpgpu_sim_wrapper::set_idle_core_power(double num_idle_core,float* idle_per_cluster) {
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_idle_core_power**********");
  fprintf(file,"\nnum_idle_core: %10.7lf",num_idle_core);

  fclose(file);

  p->sys.num_idle_cores = num_idle_core;

file = fopen("/home/pouria/Desktop/G_GPU/original_freq_per_sm/src/gpgpu-sim/idlecore.txt","a");
fprintf(file,"\nnum idle core %lf\n",num_idle_core);
  //  Per core data
  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.num_idle_cores = idle_per_cluster[i];
    fprintf(file,"%f ",  p_cores[i]->sys.num_idle_cores);
  }
  fclose(file);
  printf("\nset_idle_core_power");
  for(int i=0;i<number_shaders;i++)
    printf("\n%lf %f",p_cores[i]->sys.num_idle_cores,idle_per_cluster[i]);
  printf("\n%lf",p->sys.num_idle_cores);

  sample_perf_counters[IDLE_CORE_N] = num_idle_core;
}

void gpgpu_sim_wrapper::set_duty_cycle_power(double duty_cycle,float* duty_cycle_per_sm) {
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_pip_functions.txt","a");
  fprintf(file,"\n********set_duty_cycle_power**********");
//  fprintf(file,"\nduty_cycle: %10.7lf",duty_cycle);


  p->sys.core[0].pipeline_duty_cycle =
      duty_cycle * p->sys.scaling_coefficients[PIPE_A];

  //  Per core data
  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.core[0].pipeline_duty_cycle =
        duty_cycle_per_sm[i] * p_cores[i]->sys.scaling_coefficients[PIPE_A];
    fprintf(file,"\nduty_cycle %d: %10.7lf",i,duty_cycle_per_sm[i]);
  }
  fclose(file);
  sample_perf_counters[PIPE_A] = duty_cycle;
}

void gpgpu_sim_wrapper::set_mem_ctrl_power(double reads, double writes,
                                           double dram_precharge) {
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_mem_ctrl_power**********");
  fprintf(file,"\nreads: %10.7lf",reads);
  fprintf(file,"\nwrites: %10.7lf",writes);
  fprintf(file,"\ndram_precharge: %10.7lf",dram_precharge);

  fclose(file);


  p->sys.mc.memory_accesses = reads * p->sys.scaling_coefficients[MEM_RD] +
                              writes * p->sys.scaling_coefficients[MEM_WR];
  p->sys.mc.memory_reads = reads * p->sys.scaling_coefficients[MEM_RD];
  p->sys.mc.memory_writes = writes * p->sys.scaling_coefficients[MEM_WR];
  p->sys.mc.dram_pre = dram_precharge * p->sys.scaling_coefficients[MEM_PRE];

  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.mc.memory_accesses = 0 * p_cores[i]->sys.scaling_coefficients[MEM_RD] +
                                         0 * p_cores[i]->sys.scaling_coefficients[MEM_WR];
    p_cores[i]->sys.mc.memory_reads = 0 * p_cores[i]->sys.scaling_coefficients[MEM_RD];
    p_cores[i]->sys.mc.memory_writes = 0 * p_cores[i]->sys.scaling_coefficients[MEM_WR];
    p_cores[i]->sys.mc.dram_pre = 0 * p_cores[i]->sys.scaling_coefficients[MEM_PRE];
  }

  sample_perf_counters[MEM_RD] = reads;
  sample_perf_counters[MEM_WR] = writes;
  sample_perf_counters[MEM_PRE] = dram_precharge;
}

void gpgpu_sim_wrapper::set_exec_unit_power(double fpu_accesses,
                                            double ialu_accesses,
                                            double sfu_accesses,double *fpu_accesses_per_cluster,double *ialu_accesses_per_cluster,double *sfu_accesses_per_cluster) {

  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_exec_unit_power**********");
  fprintf(file,"\nfpu_accesses: %10.7lf",fpu_accesses);
  fprintf(file,"\nialu_accesses: %10.7lf",ialu_accesses);
  fprintf(file,"\nsfu_accesses: %10.7lf",sfu_accesses);
  fclose(file);

  p->sys.core[0].fpu_accesses =
      fpu_accesses * p->sys.scaling_coefficients[FPU_ACC];
  // Integer ALU (not present in Tesla)
  p->sys.core[0].ialu_accesses =
      ialu_accesses * p->sys.scaling_coefficients[SP_ACC];
  // Sfu accesses
  p->sys.core[0].mul_accesses =
      sfu_accesses * p->sys.scaling_coefficients[SFU_ACC];

  for(int i=0;i<number_shaders;i++)
  {
    p_cores[i]->sys.core[0].fpu_accesses =
        fpu_accesses_per_cluster[i] * p_cores[i]->sys.scaling_coefficients[FPU_ACC];
    // Integer ALU (not present in Tesla)
    // Integer ALU (not present in Tesla)
    p_cores[i]->sys.core[0].ialu_accesses =
        ialu_accesses_per_cluster[i] * p_cores[i]->sys.scaling_coefficients[SP_ACC];
    // Sfu accesses
    p_cores[i]->sys.core[0].mul_accesses =
        sfu_accesses_per_cluster[i] * p_cores[i]->sys.scaling_coefficients[SFU_ACC];
  }


  sample_perf_counters[SP_ACC] = ialu_accesses;
  sample_perf_counters[SFU_ACC] = sfu_accesses;
  sample_perf_counters[FPU_ACC] = fpu_accesses;
}

void gpgpu_sim_wrapper::set_active_lanes_power(double sp_avg_active_lane,
                                               double sfu_avg_active_lane,float * sp_avg_active_lane_per_cluster,float * sfu_avg_active_lane_per_cluster,int stat_sample_freq) {

  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_active_lanes_power**********");
  fprintf(file,"\nsp_avg_active_lane: %10.7lf",sp_avg_active_lane);
  fprintf(file,"\nsfu_avg_active_lane: %10.7lf",sfu_avg_active_lane);
  fclose(file);

  p->sys.core[0].sp_average_active_lanes = sp_avg_active_lane;
  p->sys.core[0].sfu_average_active_lanes = sfu_avg_active_lane;

  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.core[0].sp_average_active_lanes = sp_avg_active_lane_per_cluster[i];
    p_cores[i]->sys.core[0].sfu_average_active_lanes = sfu_avg_active_lane_per_cluster[i];
  }
}

void gpgpu_sim_wrapper::set_NoC_power(double noc_tot_reads,
                                      double noc_tot_writes,double*n_icnt_mem_to_simt_set_NoC_power,double *n_icnt_simt_to_mem_set_NoC_power) {
  FILE * file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Data_from_set_functions.txt","a");
  fprintf(file,"\n********set_NoC_power**********");
  fprintf(file,"\nnoc_tot_reads: %10.7lf",noc_tot_reads);
  fprintf(file,"\nnoc_tot_writes: %10.7lf",noc_tot_writes);
  fclose(file);

  p->sys.NoC[0].total_accesses =
      noc_tot_reads * p->sys.scaling_coefficients[NOC_A] +
      noc_tot_writes * p->sys.scaling_coefficients[NOC_A];

  for(int i=0;i<number_shaders;i++) {
    p_cores[i]->sys.NoC[0].total_accesses =
        n_icnt_mem_to_simt_set_NoC_power[i] *  p_cores[i]->sys.scaling_coefficients[NOC_A] +
        n_icnt_simt_to_mem_set_NoC_power[i] *  p_cores[i]->sys.scaling_coefficients[NOC_A];
  }

  sample_perf_counters[NOC_A] = noc_tot_reads + noc_tot_writes;
}

void gpgpu_sim_wrapper::power_metrics_calculations() {
  total_sample_count++;
  kernel_sample_count++;

  // Current sample power
  double sample_power =
      proc->rt_power.readOp.dynamic + sample_cmp_pwr[CONST_DYNAMICP];
    printf("\nsample_power %lf \t %lf \t %lf",sample_power,proc->rt_power.readOp.dynamic,sample_cmp_pwr[CONST_DYNAMICP]);
  double power_comm = (sum_pwr_cores-sample_power)/(number_shaders-1);
  for(int i=0;i<number_shaders;i++){
    printf("\nIsolated power per core %d %lf",i,power_per_core[i]-power_comm);
    printf("\nPer core %lf",proc_cores[i]->rt_power.readOp.dynamic);
  }
  // Average power
  // Previous + new + constant dynamic power (e.g., dynamic clocking power)
  kernel_tot_power += sample_power;
  kernel_power.avg = kernel_tot_power / kernel_sample_count;
  for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
    kernel_cmp_pwr[ind].avg += (double)sample_cmp_pwr[ind];
  }

  for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
    kernel_cmp_perf_counters[ind].avg += (double)sample_perf_counters[ind];
  }

  // Max Power
  if (sample_power > kernel_power.max) {
    kernel_power.max = sample_power;
    for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
      kernel_cmp_pwr[ind].max = (double)sample_cmp_pwr[ind];
    }
    for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
      kernel_cmp_perf_counters[ind].max = sample_perf_counters[ind];
    }
  }

  // Min Power
  if (sample_power < kernel_power.min || (kernel_power.min == 0)) {
    kernel_power.min = sample_power;
    for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
      kernel_cmp_pwr[ind].min = (double)sample_cmp_pwr[ind];
    }
    for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
      kernel_cmp_perf_counters[ind].min = sample_perf_counters[ind];
    }
  }

  gpu_tot_power.avg = (gpu_tot_power.avg + sample_power);
  gpu_tot_power.max =
      (sample_power > gpu_tot_power.max) ? sample_power : gpu_tot_power.max;
  gpu_tot_power.min =
      ((sample_power < gpu_tot_power.min) || (gpu_tot_power.min == 0))
          ? sample_power
          : gpu_tot_power.min;
}

void gpgpu_sim_wrapper::print_trace_files() {
  open_files();

  for (unsigned i = 0; i < num_perf_counters; ++i) {
    gzprintf(metric_trace_file, "%f,", sample_perf_counters[i]);
  }
  gzprintf(metric_trace_file, "\n");

  gzprintf(power_trace_file, "%f,", proc_power);
  for (unsigned i = 0; i < num_pwr_cmps; ++i) {
    gzprintf(power_trace_file, "%f,", sample_cmp_pwr[i]);
  }
  gzprintf(power_trace_file, "\n");

  close_files();

  FILE* file;
  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Performance_counters.txt","a");
  fprintf(file,"\n********************ROUND********************");
  for (unsigned i = 0; i < num_perf_counters; ++i) {
    if(i %(int)(num_perf_counters/4) == 0)
      fprintf(file,"\n");
    fprintf(file, "%f,", sample_perf_counters[i]);
  }

}

void gpgpu_sim_wrapper::update_coefficients() {
  initpower_coeff[FP_INT] = proc->cores[0]->get_coefficient_fpint_insts();
  effpower_coeff[FP_INT] =
      initpower_coeff[FP_INT] * p->sys.scaling_coefficients[FP_INT];

  initpower_coeff[TOT_INST] = proc->cores[0]->get_coefficient_tot_insts();
  effpower_coeff[TOT_INST] =
      initpower_coeff[TOT_INST] * p->sys.scaling_coefficients[TOT_INST];

  initpower_coeff[REG_RD] =
      proc->cores[0]->get_coefficient_regreads_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  initpower_coeff[REG_WR] =
      proc->cores[0]->get_coefficient_regwrites_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  initpower_coeff[NON_REG_OPs] =
      proc->cores[0]->get_coefficient_noregfileops_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  effpower_coeff[REG_RD] =
      initpower_coeff[REG_RD] * p->sys.scaling_coefficients[REG_RD];
  effpower_coeff[REG_WR] =
      initpower_coeff[REG_WR] * p->sys.scaling_coefficients[REG_WR];
  effpower_coeff[NON_REG_OPs] =
      initpower_coeff[NON_REG_OPs] * p->sys.scaling_coefficients[NON_REG_OPs];

  initpower_coeff[IC_H] = proc->cores[0]->get_coefficient_icache_hits();
  initpower_coeff[IC_M] = proc->cores[0]->get_coefficient_icache_misses();
  effpower_coeff[IC_H] =
      initpower_coeff[IC_H] * p->sys.scaling_coefficients[IC_H];
  effpower_coeff[IC_M] =
      initpower_coeff[IC_M] * p->sys.scaling_coefficients[IC_M];

  initpower_coeff[CC_H] = (proc->cores[0]->get_coefficient_ccache_readhits() +
                           proc->get_coefficient_readcoalescing());
  initpower_coeff[CC_M] = (proc->cores[0]->get_coefficient_ccache_readmisses() +
                           proc->get_coefficient_readcoalescing());
  effpower_coeff[CC_H] =
      initpower_coeff[CC_H] * p->sys.scaling_coefficients[CC_H];
  effpower_coeff[CC_M] =
      initpower_coeff[CC_M] * p->sys.scaling_coefficients[CC_M];

  initpower_coeff[TC_H] = (proc->cores[0]->get_coefficient_tcache_readhits() +
                           proc->get_coefficient_readcoalescing());
  initpower_coeff[TC_M] = (proc->cores[0]->get_coefficient_tcache_readmisses() +
                           proc->get_coefficient_readcoalescing());
  effpower_coeff[TC_H] =
      initpower_coeff[TC_H] * p->sys.scaling_coefficients[TC_H];
  effpower_coeff[TC_M] =
      initpower_coeff[TC_M] * p->sys.scaling_coefficients[TC_M];

  initpower_coeff[SHRD_ACC] =
      proc->cores[0]->get_coefficient_sharedmemory_readhits();
  effpower_coeff[SHRD_ACC] =
      initpower_coeff[SHRD_ACC] * p->sys.scaling_coefficients[SHRD_ACC];

  initpower_coeff[DC_RH] = (proc->cores[0]->get_coefficient_dcache_readhits() +
                            proc->get_coefficient_readcoalescing());
  initpower_coeff[DC_RM] =
      (proc->cores[0]->get_coefficient_dcache_readmisses() +
       proc->get_coefficient_readcoalescing());
  initpower_coeff[DC_WH] = (proc->cores[0]->get_coefficient_dcache_writehits() +
                            proc->get_coefficient_writecoalescing());
  initpower_coeff[DC_WM] =
      (proc->cores[0]->get_coefficient_dcache_writemisses() +
       proc->get_coefficient_writecoalescing());
  effpower_coeff[DC_RH] =
      initpower_coeff[DC_RH] * p->sys.scaling_coefficients[DC_RH];
  effpower_coeff[DC_RM] =
      initpower_coeff[DC_RM] * p->sys.scaling_coefficients[DC_RM];
  effpower_coeff[DC_WH] =
      initpower_coeff[DC_WH] * p->sys.scaling_coefficients[DC_WH];
  effpower_coeff[DC_WM] =
      initpower_coeff[DC_WM] * p->sys.scaling_coefficients[DC_WM];

  initpower_coeff[L2_RH] = proc->get_coefficient_l2_read_hits();
  initpower_coeff[L2_RM] = proc->get_coefficient_l2_read_misses();
  initpower_coeff[L2_WH] = proc->get_coefficient_l2_write_hits();
  initpower_coeff[L2_WM] = proc->get_coefficient_l2_write_misses();
  effpower_coeff[L2_RH] =
      initpower_coeff[L2_RH] * p->sys.scaling_coefficients[L2_RH];
  effpower_coeff[L2_RM] =
      initpower_coeff[L2_RM] * p->sys.scaling_coefficients[L2_RM];
  effpower_coeff[L2_WH] =
      initpower_coeff[L2_WH] * p->sys.scaling_coefficients[L2_WH];
  effpower_coeff[L2_WM] =
      initpower_coeff[L2_WM] * p->sys.scaling_coefficients[L2_WM];

  initpower_coeff[IDLE_CORE_N] =
      p->sys.idle_core_power * proc->cores[0]->executionTime;
  effpower_coeff[IDLE_CORE_N] =
      initpower_coeff[IDLE_CORE_N] * p->sys.scaling_coefficients[IDLE_CORE_N];

  initpower_coeff[PIPE_A] = proc->cores[0]->get_coefficient_duty_cycle();
  effpower_coeff[PIPE_A] =
      initpower_coeff[PIPE_A] * p->sys.scaling_coefficients[PIPE_A];

  initpower_coeff[MEM_RD] = proc->get_coefficient_mem_reads();
  initpower_coeff[MEM_WR] = proc->get_coefficient_mem_writes();
  initpower_coeff[MEM_PRE] = proc->get_coefficient_mem_pre();
  effpower_coeff[MEM_RD] =
      initpower_coeff[MEM_RD] * p->sys.scaling_coefficients[MEM_RD];
  effpower_coeff[MEM_WR] =
      initpower_coeff[MEM_WR] * p->sys.scaling_coefficients[MEM_WR];
  effpower_coeff[MEM_PRE] =
      initpower_coeff[MEM_PRE] * p->sys.scaling_coefficients[MEM_PRE];

  initpower_coeff[SP_ACC] =
      proc->cores[0]->get_coefficient_ialu_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  ;
  initpower_coeff[SFU_ACC] = proc->cores[0]->get_coefficient_sfu_accesses();
  initpower_coeff[FPU_ACC] = proc->cores[0]->get_coefficient_fpu_accesses();

  effpower_coeff[SP_ACC] =
      initpower_coeff[SP_ACC] * p->sys.scaling_coefficients[SP_ACC];
  effpower_coeff[SFU_ACC] =
      initpower_coeff[SFU_ACC] * p->sys.scaling_coefficients[SFU_ACC];
  effpower_coeff[FPU_ACC] =
      initpower_coeff[FPU_ACC] * p->sys.scaling_coefficients[FPU_ACC];

  initpower_coeff[NOC_A] = proc->get_coefficient_noc_accesses();
  effpower_coeff[NOC_A] =
      initpower_coeff[NOC_A] * p->sys.scaling_coefficients[NOC_A];

  const_dynamic_power =
      proc->get_const_dynamic_power() / (proc->cores[0]->executionTime);

  for (unsigned i = 0; i < num_perf_counters; i++) {
    initpower_coeff[i] /= (proc->cores[0]->executionTime);
    effpower_coeff[i] /= (proc->cores[0]->executionTime);
  }
}

void gpgpu_sim_wrapper::update_coefficients_per_core() {
  for (int i = 0; i < number_shaders; i++) {
    initpower_coeff[FP_INT] = proc_cores[i]->cores[0]->get_coefficient_fpint_insts();
    effpower_coeff[FP_INT] =
        initpower_coeff[FP_INT] * p->sys.scaling_coefficients[FP_INT];

    initpower_coeff[TOT_INST] = proc_cores[i]->cores[0]->get_coefficient_tot_insts();
    effpower_coeff[TOT_INST] =
        initpower_coeff[TOT_INST] * p->sys.scaling_coefficients[TOT_INST];

    initpower_coeff[REG_RD] =
        proc_cores[i]->cores[0]->get_coefficient_regreads_accesses() *
        (proc_cores[i]->cores[0]->exu->rf_fu_clockRate / proc_cores[i]->cores[0]->exu->clockRate);
    initpower_coeff[REG_WR] =
        proc_cores[i]->cores[0]->get_coefficient_regwrites_accesses() *
        (proc_cores[i]->cores[0]->exu->rf_fu_clockRate / proc_cores[i]->cores[0]->exu->clockRate);
    initpower_coeff[NON_REG_OPs] =
        proc_cores[i]->cores[0]->get_coefficient_noregfileops_accesses() *
        (proc_cores[i]->cores[0]->exu->rf_fu_clockRate / proc_cores[i]->cores[0]->exu->clockRate);
    effpower_coeff[REG_RD] =
        initpower_coeff[REG_RD] * p->sys.scaling_coefficients[REG_RD];
    effpower_coeff[REG_WR] =
        initpower_coeff[REG_WR] * p->sys.scaling_coefficients[REG_WR];
    effpower_coeff[NON_REG_OPs] =
        initpower_coeff[NON_REG_OPs] * p->sys.scaling_coefficients[NON_REG_OPs];

    initpower_coeff[IC_H] = proc_cores[i]->cores[0]->get_coefficient_icache_hits();
    initpower_coeff[IC_M] = proc_cores[i]->cores[0]->get_coefficient_icache_misses();
    effpower_coeff[IC_H] =
        initpower_coeff[IC_H] * p->sys.scaling_coefficients[IC_H];
    effpower_coeff[IC_M] =
        initpower_coeff[IC_M] * p->sys.scaling_coefficients[IC_M];

    initpower_coeff[CC_H] = (proc_cores[i]->cores[0]->get_coefficient_ccache_readhits() +
                             proc_cores[i]->get_coefficient_readcoalescing());
    initpower_coeff[CC_M] =
        (proc_cores[i]->cores[0]->get_coefficient_ccache_readmisses() +
         proc_cores[i]->get_coefficient_readcoalescing());
    effpower_coeff[CC_H] =
        initpower_coeff[CC_H] * p->sys.scaling_coefficients[CC_H];
    effpower_coeff[CC_M] =
        initpower_coeff[CC_M] * p->sys.scaling_coefficients[CC_M];

    initpower_coeff[TC_H] = (proc_cores[i]->cores[0]->get_coefficient_tcache_readhits() +
                             proc_cores[i]->get_coefficient_readcoalescing());
    initpower_coeff[TC_M] =
        (proc_cores[i]->cores[0]->get_coefficient_tcache_readmisses() +
         proc_cores[i]->get_coefficient_readcoalescing());
    effpower_coeff[TC_H] =
        initpower_coeff[TC_H] * p->sys.scaling_coefficients[TC_H];
    effpower_coeff[TC_M] =
        initpower_coeff[TC_M] * p->sys.scaling_coefficients[TC_M];

    initpower_coeff[SHRD_ACC] =
        proc_cores[i]->cores[0]->get_coefficient_sharedmemory_readhits();
    effpower_coeff[SHRD_ACC] =
        initpower_coeff[SHRD_ACC] * p->sys.scaling_coefficients[SHRD_ACC];

    initpower_coeff[DC_RH] =
        (proc_cores[i]->cores[0]->get_coefficient_dcache_readhits() +
         proc_cores[i]->get_coefficient_readcoalescing());
    initpower_coeff[DC_RM] =
        (proc_cores[i]->cores[0]->get_coefficient_dcache_readmisses() +
         proc_cores[i]->get_coefficient_readcoalescing());
    initpower_coeff[DC_WH] =
        (proc_cores[i]->cores[0]->get_coefficient_dcache_writehits() +
         proc_cores[i]->get_coefficient_writecoalescing());
    initpower_coeff[DC_WM] =
        (proc_cores[i]->cores[0]->get_coefficient_dcache_writemisses() +
         proc_cores[i]->get_coefficient_writecoalescing());
    effpower_coeff[DC_RH] =
        initpower_coeff[DC_RH] * p->sys.scaling_coefficients[DC_RH];
    effpower_coeff[DC_RM] =
        initpower_coeff[DC_RM] * p->sys.scaling_coefficients[DC_RM];
    effpower_coeff[DC_WH] =
        initpower_coeff[DC_WH] * p->sys.scaling_coefficients[DC_WH];
    effpower_coeff[DC_WM] =
        initpower_coeff[DC_WM] * p->sys.scaling_coefficients[DC_WM];

    initpower_coeff[L2_RH] = proc_cores[i]->get_coefficient_l2_read_hits();
    initpower_coeff[L2_RM] = proc_cores[i]->get_coefficient_l2_read_misses();
    initpower_coeff[L2_WH] = proc_cores[i]->get_coefficient_l2_write_hits();
    initpower_coeff[L2_WM] = proc_cores[i]->get_coefficient_l2_write_misses();
    effpower_coeff[L2_RH] =
        initpower_coeff[L2_RH] * p->sys.scaling_coefficients[L2_RH];
    effpower_coeff[L2_RM] =
        initpower_coeff[L2_RM] * p->sys.scaling_coefficients[L2_RM];
    effpower_coeff[L2_WH] =
        initpower_coeff[L2_WH] * p->sys.scaling_coefficients[L2_WH];
    effpower_coeff[L2_WM] =
        initpower_coeff[L2_WM] * p->sys.scaling_coefficients[L2_WM];

    initpower_coeff[IDLE_CORE_N] =
        p->sys.idle_core_power * proc_cores[i]->cores[0]->executionTime;
    effpower_coeff[IDLE_CORE_N] =
        initpower_coeff[IDLE_CORE_N] * p->sys.scaling_coefficients[IDLE_CORE_N];

    initpower_coeff[PIPE_A] = proc_cores[i]->cores[0]->get_coefficient_duty_cycle();
    effpower_coeff[PIPE_A] =
        initpower_coeff[PIPE_A] * p->sys.scaling_coefficients[PIPE_A];

    initpower_coeff[MEM_RD] = proc_cores[i]->get_coefficient_mem_reads();
    initpower_coeff[MEM_WR] = proc_cores[i]->get_coefficient_mem_writes();
    initpower_coeff[MEM_PRE] = proc_cores[i]->get_coefficient_mem_pre();
    effpower_coeff[MEM_RD] =
        initpower_coeff[MEM_RD] * p->sys.scaling_coefficients[MEM_RD];
    effpower_coeff[MEM_WR] =
        initpower_coeff[MEM_WR] * p->sys.scaling_coefficients[MEM_WR];
    effpower_coeff[MEM_PRE] =
        initpower_coeff[MEM_PRE] * p->sys.scaling_coefficients[MEM_PRE];

    initpower_coeff[SP_ACC] =
        proc_cores[i]->cores[0]->get_coefficient_ialu_accesses() *
        (proc_cores[i]->cores[0]->exu->rf_fu_clockRate / proc_cores[i]->cores[0]->exu->clockRate);
    ;
    initpower_coeff[SFU_ACC] = proc_cores[i]->cores[0]->get_coefficient_sfu_accesses();
    initpower_coeff[FPU_ACC] = proc_cores[i]->cores[0]->get_coefficient_fpu_accesses();

    effpower_coeff[SP_ACC] =
        initpower_coeff[SP_ACC] * p->sys.scaling_coefficients[SP_ACC];
    effpower_coeff[SFU_ACC] =
        initpower_coeff[SFU_ACC] * p->sys.scaling_coefficients[SFU_ACC];
    effpower_coeff[FPU_ACC] =
        initpower_coeff[FPU_ACC] * p->sys.scaling_coefficients[FPU_ACC];

    initpower_coeff[NOC_A] = proc_cores[i]->get_coefficient_noc_accesses();
    effpower_coeff[NOC_A] =
        initpower_coeff[NOC_A] * p->sys.scaling_coefficients[NOC_A];

    const_dynamic_power =
        proc_cores[i]->get_const_dynamic_power() / (proc_cores[i]->cores[0]->executionTime);


    for (unsigned i = 0; i < num_perf_counters; i++) {
      initpower_coeff[i] /= (proc_cores[i]->cores[0]->executionTime);
      effpower_coeff[i] /= (proc_cores[i]->cores[0]->executionTime);
    }
  }
}

void gpgpu_sim_wrapper::smp_cpm_pwr_print(){
  for(int i=0;i<number_shaders;i++){
    printf("\nTotal component %d: %lf",i,sample_cmp_pwr[i]);
    for(int j=0;j<num_pwr_cmps;j++){
      printf("\ncore %d component %d: %lf",i,j,sample_cmp_pwr_per_core[i][j]);
    }
  }

}
void gpgpu_sim_wrapper::update_components_power_per_core(bool loop,double base_freq) {
  //  update_coefficients_per_core();

  sample_cmp_pwr[IBP] = 0;
  sample_cmp_pwr[SHRDP] = 0;
  sample_cmp_pwr[RFP] = 0;
  sample_cmp_pwr[SPP] = 0;
  sample_cmp_pwr[SFUP] = 0;
  sample_cmp_pwr[FPUP] = 0;
  sample_cmp_pwr[SCHEDP] = 0;
  sample_cmp_pwr[NOCP] = 0;
  sample_cmp_pwr[PIPEP] = 0;
  sample_cmp_pwr[IDLE_COREP] = 0;

  FILE *file = fopen("/home/pouria/Desktop/G_GPU/DATA/PIPEP.txt","a");
  FILE *file_pip;
  file_pip = fopen("/home/pouria/Desktop/G_GPU/DATA/pip_energy.txt","a");
  fprintf(file_pip,"\n***********************");
  FILE* IBP_ = fopen("/home/pouria/Desktop/G_GPU/DATA/IBP.txt","a");
  fprintf(IBP_,"\n***********************");
  for (int i = 0; i < number_shaders; i++) {
    sample_cmp_pwr[IBP] +=
        (proc_cores[i]->cores[0]->ifu->IB->rt_power.readOp.dynamic +
         proc_cores[i]->cores[0]->ifu->IB->rt_power.writeOp.dynamic +
         proc_cores[i]->cores[0]->ifu->ID_misc->rt_power.readOp.dynamic +
         proc_cores[i]->cores[0]->ifu->ID_operand->rt_power.readOp.dynamic +
         proc_cores[i]->cores[0]->ifu->ID_inst->rt_power.readOp.dynamic) /
        (proc_cores[i]->cores[0]->executionTime);

    fprintf(IBP_,"\n%d %2.12lf %2.10lf %2.12lf",i,(proc_cores[i]->cores[0]->ifu->IB->rt_power.readOp.dynamic +
                                  proc_cores[i]->cores[0]->ifu->IB->rt_power.writeOp.dynamic +
                                  proc_cores[i]->cores[0]->ifu->ID_misc->rt_power.readOp.dynamic +
                                  proc_cores[i]->cores[0]->ifu->ID_operand->rt_power.readOp.dynamic +
                                  proc_cores[i]->cores[0]->ifu->ID_inst->rt_power.readOp.dynamic),
            (proc_cores[i]->cores[0]->ifu->IB->rt_power.readOp.dynamic +
             proc_cores[i]->cores[0]->ifu->IB->rt_power.writeOp.dynamic +
             proc_cores[i]->cores[0]->ifu->ID_misc->rt_power.readOp.dynamic +
             proc_cores[i]->cores[0]->ifu->ID_operand->rt_power.readOp.dynamic +
             proc_cores[i]->cores[0]->ifu->ID_inst->rt_power.readOp.dynamic)/(proc_cores[i]->cores[0]->executionTime),(proc_cores[i]->cores[0]->executionTime));

    sample_cmp_pwr[SHRDP] +=
        proc_cores[i]->cores[0]->lsu->sharedmemory.rt_power.readOp.dynamic /
        (proc_cores[i]->cores[0]->executionTime);

    sample_cmp_pwr[RFP] +=
        (proc_cores[i]->cores[0]->exu->rfu->rt_power.readOp.dynamic /
         (proc_cores[i]->cores[0]->executionTime)) *
        (proc_cores[i]->cores[0]->exu->rf_fu_clockRate /
         proc_cores[i]->cores[0]->exu->clockRate);

    sample_cmp_pwr[SPP] +=
        (proc_cores[i]->cores[0]->exu->exeu->rt_power.readOp.dynamic /
         (proc_cores[i]->cores[0]->executionTime)) *
        (proc_cores[i]->cores[0]->exu->rf_fu_clockRate /
         proc_cores[i]->cores[0]->exu->clockRate);

    sample_cmp_pwr[SFUP] +=
        (proc_cores[i]->cores[0]->exu->mul->rt_power.readOp.dynamic /
         (proc_cores[i]->cores[0]->executionTime));

    sample_cmp_pwr[FPUP] +=
        (proc_cores[i]->cores[0]->exu->fp_u->rt_power.readOp.dynamic /
         (proc_cores[i]->cores[0]->executionTime));

    sample_cmp_pwr[SCHEDP] +=
        proc_cores[i]->cores[0]->exu->scheu->rt_power.readOp.dynamic /
        (proc_cores[i]->cores[0]->executionTime);

    sample_cmp_pwr[NOCP] +=
        proc_cores[i]->nocs[0]->rt_power.readOp.dynamic /
        (proc_cores[i]->cores[0]->executionTime);

    sample_cmp_pwr[PIPEP] +=
        proc_cores[i]->cores[0]->Pipeline_energy /
        (proc->cores[0]->executionTime);

    FILE * file_final_ifu;
    file_final_ifu = fopen("/home/pouria/Desktop/G_GPU/DATA/CORE_PIP_ifu.txt","a");
      fprintf(file_final_ifu,"\n%d: %6.10lf %lf %2.10lf",i, proc_cores[i]->cores[0]->Pipeline_energy,proc_cores[i]->cores[0]->Pipeline_energy /
                                                                proc->cores[0]->executionTime,proc->cores[0]->executionTime);
    fflush(file_final_ifu);
    fclose(file_final_ifu);

if(base_freq == 700000000)
    fprintf(file_pip,"\n%d: %lf",i,proc_cores[i]->cores[0]->Pipeline_energy /
                                          (proc->cores[0]->executionTime)/15);

    sample_cmp_pwr[IDLE_COREP] +=
        proc_cores[i]->cores[0]->IdleCoreEnergy /
        (proc_cores[i]->cores[0]->executionTime);
  }
  fprintf(file_pip,"\n%lf",sample_cmp_pwr[PIPEP]/15);
  fclose(file_pip);
  fclose(IBP_);
  sample_cmp_pwr[PIPEP] /= number_shaders;

  file = fopen("/home/pouria/Desktop/G_GPU/DATA/Components_power.txt","a");
  fprintf(file,"\n*****************ROUND***************");
  for(int i=0;i<num_pwr_cmps;i++) {
    fprintf(file, "\n%s: %2.7lf ", pwr_cmp_label[i], sample_cmp_pwr[i]);
  }
  fprintf(file,"\n");

  fclose(file);

}
void gpgpu_sim_wrapper::update_components_power(int x) {
  if(x)
    update_coefficients();
  proc_power = proc->rt_power.readOp.dynamic;

  sample_cmp_pwr[ICP] = proc->cores[0]->ifu->icache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[DCP] = proc->cores[0]->lsu->dcache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[TCP] = proc->cores[0]->lsu->tcache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[CCP] = proc->cores[0]->lsu->ccache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);


  sample_cmp_pwr[L2CP] = (proc->XML->sys.number_of_L2s > 0)
                             ? proc->l2array[0]->rt_power.readOp.dynamic /
                                   (proc->cores[0]->executionTime)
                             : 0;

  sample_cmp_pwr[MCP] = (proc->mc->rt_power.readOp.dynamic -
                         proc->mc->dram->rt_power.readOp.dynamic) /
                        (proc->cores[0]->executionTime);



  sample_cmp_pwr[DRAMP] =
      proc->mc->dram->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);



  // This constant dynamic power (e.g., clock power) part is estimated via
  // regression model.
  sample_cmp_pwr[CONST_DYNAMICP] = 0;
  double cnst_dyn =
      proc->get_const_dynamic_power() / (proc->cores[0]->executionTime);
  // If the regression scaling term is greater than the recorded constant
  // dynamic power then use the difference (other portion already added to
  // dynamic power). Else, all the constant dynamic power is accounted for, add
  // nothing.
  if (p->sys.scaling_coefficients[CONST_DYNAMICN] > cnst_dyn)
    sample_cmp_pwr[CONST_DYNAMICP] =
        (p->sys.scaling_coefficients[CONST_DYNAMICN] - cnst_dyn);


//  bool check = false;
//
//  check = sanity_check(sum_pwr_cmp, proc_power);

//  assert("Total Power does not equal the sum of the components\n" && (check));
}

//double gpgpu_sim_wrapper::sum_per_sm_and_shard_power(double* cluster_freq){
//  double Total_power=0;
//  FILE * file;
//  file =  fopen("/home/pouria/Desktop/G_GPU/DATA/component_power.txt","a");
//  fprintf(file,"\n*****************ROUND***************\n%lf",cluster_freq[0]);
//  for(int i=0;i<num_pwr_cmps;i++) {
//    Total_power += sample_cmp_pwr[i];
//    printf("\n%d: %lf",i,sample_cmp_pwr[i]);
//      fprintf(file, "\n%s: %2.7lf", pwr_cmp_label[i], sample_cmp_pwr[i]);
//  }
//  fclose(file);
//  return Total_power;
//}

double gpgpu_sim_wrapper::sum_per_sm_and_shard_power(double* cluster_freq) {
  double Power = 0;
  for (int i = 0; i < number_shaders; i++) {
    sample_cmp_pwr_S[i] = 0;
  }
  FILE * file_final;
  file_final = fopen("/home/pouria/Desktop/G_GPU/DATA/FINAL.txt","a");
  FILE * file_final_power;
  file_final_power = fopen("/home/pouria/Desktop/G_GPU/DATA/FINAL_POWER.txt","a");
  sample_cmp_pwr_const = 0;
  sample_cmp_pwr_Shrd = 0;
  for (int i = 0; i < number_shaders; i++) {

           sample_cmp_pwr_S[i] += (proc_cores[i]->cores[0]->ifu->IB->rt_power.readOp.dynamic +
                              proc_cores[i]->cores[0]->ifu->IB->rt_power.writeOp.dynamic +
                              proc_cores[i]->cores[0]->ifu->ID_misc->rt_power.readOp.dynamic +
                              proc_cores[i]->cores[0]->ifu->ID_operand->rt_power.readOp.dynamic +
                              proc_cores[i]->cores[0]->ifu->ID_inst->rt_power.readOp.dynamic) /
                             (proc_cores[i]->cores[0]->executionTime);


           sample_cmp_pwr_S[i] +=
               (proc_cores[i]->cores[0]->exu->rfu->rt_power.readOp.dynamic /
            (proc_cores[i]->cores[0]->executionTime)) *
               (proc_cores[i]->cores[0]->exu->rf_fu_clockRate /
                proc_cores[i]->cores[0]->exu->clockRate);


           sample_cmp_pwr_S[i] += proc_cores[i]->cores[0]->lsu->sharedmemory.rt_power.readOp.dynamic /
               (proc_cores[i]->cores[0]->executionTime);



           sample_cmp_pwr_S[i] +=
               (proc_cores[i]->cores[0]->exu->exeu->rt_power.readOp.dynamic /
                (proc_cores[i]->cores[0]->executionTime)) *
               (proc_cores[i]->cores[0]->exu->rf_fu_clockRate /
                proc_cores[i]->cores[0]->exu->clockRate);


           sample_cmp_pwr_S[i] +=
               (proc_cores[i]->cores[0]->exu->mul->rt_power.readOp.dynamic /
                (proc_cores[i]->cores[0]->executionTime));


           sample_cmp_pwr_S[i] +=
               (proc_cores[i]->cores[0]->exu->fp_u->rt_power.readOp.dynamic /
                (proc_cores[i]->cores[0]->executionTime));


           sample_cmp_pwr_S[i] +=
               proc_cores[i]->cores[0]->exu->scheu->rt_power.readOp.dynamic /
               (proc_cores[i]->cores[0]->executionTime);


           sample_cmp_pwr_S[i] +=
               proc_cores[i]->nocs[0]->rt_power.readOp.dynamic /
               (proc_cores[i]->cores[0]->executionTime);


           sample_cmp_pwr_S[i] +=
               proc_cores[i]->cores[0]->Pipeline_energy /
               (proc->cores[0]->executionTime)/number_shaders;


           sample_cmp_pwr_const +=
               proc_cores[i]->cores[0]->IdleCoreEnergy /
               (proc_cores[i]->cores[0]->executionTime);
           Power +=  sample_cmp_pwr_S[i];
//           fprintf(file_final,"\nsample_cmp_pwr_S[%d]: %lf",i,sample_cmp_pwr_S[i]);

    }


//    fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf %10.10lf %10.10lf",
//            sample_cmp_pwr_Shrd,proc->cores[0]->ifu->icache.rt_power.readOp.dynamic,proc->cores[0]->executionTime);

  sample_cmp_pwr_Shrd += proc->cores[0]->ifu->icache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

//  fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf %10.10lf %10.10lf",
//          sample_cmp_pwr_Shrd,proc->cores[0]->lsu->dcache.rt_power.readOp.dynamic,proc->cores[0]->executionTime);

  sample_cmp_pwr_Shrd += proc->cores[0]->lsu->dcache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

//  fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf %10.10lf %10.10lf",
//          sample_cmp_pwr_Shrd,proc->cores[0]->lsu->tcache.rt_power.readOp.dynamic,proc->cores[0]->executionTime);

  sample_cmp_pwr_Shrd += proc->cores[0]->lsu->tcache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

//  fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf %10.10lf %10.10lf",
//          sample_cmp_pwr_Shrd,proc->cores[0]->lsu->ccache.rt_power.readOp.dynamic,proc->cores[0]->executionTime);

  sample_cmp_pwr_Shrd += proc->cores[0]->lsu->ccache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

//  fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf %10.10lf %10.10lf",
//          sample_cmp_pwr_Shrd,proc->l2array[0]->rt_power.readOp.dynamic,proc->cores[0]->executionTime);

  sample_cmp_pwr_Shrd += (proc->XML->sys.number_of_L2s > 0)
                             ? proc->l2array[0]->rt_power.readOp.dynamic /
                                   (proc->cores[0]->executionTime)
                             : 0;
//  fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf %10.10lf %10.10lf",
//          sample_cmp_pwr_Shrd,(proc->mc->rt_power.readOp.dynamic -
//                                proc->mc->dram->rt_power.readOp.dynamic),proc->cores[0]->executionTime);

  sample_cmp_pwr_Shrd += (proc->mc->rt_power.readOp.dynamic -
                         proc->mc->dram->rt_power.readOp.dynamic) /
                        (proc->cores[0]->executionTime);


//  fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf %10.10lf %10.10lf",
//          sample_cmp_pwr_Shrd,proc->mc->rt_power.readOp.dynamic,proc->cores[0]->executionTime);

  sample_cmp_pwr_Shrd +=
      proc->mc->dram->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);

//  fprintf(file_final,"\nsample_cmp_pwr_Shrd: %lf",sample_cmp_pwr_Shrd);


  // This constant dynamic power (e.g., clock power) part is estimated via
  // regression model.
  sample_cmp_pwr[CONST_DYNAMICP] = 0;
  double cnst_dyn =
      proc->get_const_dynamic_power() / (proc->cores[0]->executionTime);
  // If the regression scaling term is greater than the recorded constant
  // dynamic power then use the difference (other portion already added to
  // dynamic power). Else, all the constant dynamic power is accounted for, add
  // nothing.
  if (p->sys.scaling_coefficients[CONST_DYNAMICN] > cnst_dyn)
    sample_cmp_pwr_const +=
        (p->sys.scaling_coefficients[CONST_DYNAMICN] - cnst_dyn);
  fprintf(file_final,"\nsample_cmp_pwr_const: %lf sample_cmp_pwr_Shrd: %lf sample_cmp_pwr_S: %lf power:\t %lf",
          sample_cmp_pwr_const,sample_cmp_pwr_Shrd,Power,Power + sample_cmp_pwr_Shrd + sample_cmp_pwr_const);
  Power += sample_cmp_pwr_Shrd + sample_cmp_pwr_const;
  fprintf(file_final_power,"\n+ %lf",Power);


//  fprintf(file_final,"\n**************\n");
  fflush(file_final);
  fclose(file_final);
  fclose(file_final_power);
  FILE* file_throughput = fopen("/home/pouria/Desktop/G_GPU/DATA/throughput.txt","a");
  fprintf(file_throughput,"\n*************\n");
  for(int i=0;i<number_shaders;i++)
  fprintf(file_throughput,"\n%d %lf",i,Throughput[i]);
  fclose(file_throughput);
  return Power;
}
void gpgpu_sim_wrapper::compute(bool loop) {
  for(int i=0;i<number_shaders;i++){
    printf("\nCore %d",i);
    proc_cores[i]->compute(loop);
  }


  proc->compute(loop);
}
void gpgpu_sim_wrapper::print_power_kernel_stats(
    double gpu_sim_cycle, double gpu_tot_sim_cycle, double init_value,
    const std::string& kernel_info_string, bool print_trace) {
  detect_print_steady_state(1, init_value);
  if (g_power_simulation_enabled) {
    powerfile << kernel_info_string << std::endl;

    sanity_check((kernel_power.avg * kernel_sample_count), kernel_tot_power);
    powerfile << "Kernel Average Power Data:" << std::endl;
    powerfile << "kernel_avg_power = " << kernel_power.avg << std::endl;

    for (unsigned i = 0; i < num_pwr_cmps; ++i) {
      powerfile << "gpu_avg_" << pwr_cmp_label[i] << " = "
                << kernel_cmp_pwr[i].avg / kernel_sample_count << std::endl;
    }
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      powerfile << "gpu_avg_" << perf_count_label[i] << " = "
                << kernel_cmp_perf_counters[i].avg / kernel_sample_count
                << std::endl;
    }

    powerfile << std::endl << "Kernel Maximum Power Data:" << std::endl;
    powerfile << "kernel_max_power = " << kernel_power.max << std::endl;
    for (unsigned i = 0; i < num_pwr_cmps; ++i) {
      powerfile << "gpu_max_" << pwr_cmp_label[i] << " = "
                << kernel_cmp_pwr[i].max << std::endl;
    }
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      powerfile << "gpu_max_" << perf_count_label[i] << " = "
                << kernel_cmp_perf_counters[i].max << std::endl;
    }

    powerfile << std::endl << "Kernel Minimum Power Data:" << std::endl;
    powerfile << "kernel_min_power = " << kernel_power.min << std::endl;
    for (unsigned i = 0; i < num_pwr_cmps; ++i) {
      powerfile << "gpu_min_" << pwr_cmp_label[i] << " = "
                << kernel_cmp_pwr[i].min << std::endl;
    }
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      powerfile << "gpu_min_" << perf_count_label[i] << " = "
                << kernel_cmp_perf_counters[i].min << std::endl;
    }

    powerfile << std::endl
              << "Accumulative Power Statistics Over Previous Kernels:"
              << std::endl;
    powerfile << "gpu_tot_avg_power = "
              << gpu_tot_power.avg / total_sample_count << std::endl;
    powerfile << "gpu_tot_max_power = " << gpu_tot_power.max << std::endl;
    powerfile << "gpu_tot_min_power = " << gpu_tot_power.min << std::endl;
    powerfile << std::endl << std::endl;
    powerfile.flush();

    if (print_trace) {
      print_trace_files();
    }
  }
}
void gpgpu_sim_wrapper::dump() {
  if (g_power_per_cycle_dump) proc->displayEnergy(2, 5);
}

void gpgpu_sim_wrapper::print_steady_state(int position, double init_val) {
  double temp_avg = sample_val / (double)samples.size();
  double temp_ipc = (init_val - init_inst_val) /
                    (double)(samples.size() * gpu_stat_sample_freq);

  if ((samples.size() >
       gpu_steady_min_period)) {  // If steady state occurred for some time,
                                  // print to file
    has_written_avg = true;
    gzprintf(steady_state_tacking_file, "%u,%d,%f,%f,", sample_start,
             total_sample_count, temp_avg, temp_ipc);
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      gzprintf(steady_state_tacking_file, "%f,",
               samples_counter.at(i) / ((double)samples.size()));
    }
    gzprintf(steady_state_tacking_file, "\n");
  } else {
    if (!has_written_avg && position)
      gzprintf(steady_state_tacking_file,
               "ERROR! Not enough steady state points to generate average\n");
  }

  sample_start = 0;
  sample_val = 0;
  init_inst_val = init_val;
  samples.clear();
  samples_counter.clear();
  pwr_counter.clear();
  assert(samples.size() == 0);
}

void gpgpu_sim_wrapper::detect_print_steady_state(int position,
                                                  double init_val) {
  // Calculating Average
  if (g_power_simulation_enabled && g_steady_power_levels_enabled) {
    steady_state_tacking_file = gzopen(g_steady_state_tracking_filename, "a");
    if (position == 0) {
      if (samples.size() == 0) {
        // First sample
        sample_start = total_sample_count;
        sample_val = proc->rt_power.readOp.dynamic;
        init_inst_val = init_val;
        samples.push_back(proc->rt_power.readOp.dynamic);
        assert(samples_counter.size() == 0);
        assert(pwr_counter.size() == 0);

        for (unsigned i = 0; i < (num_perf_counters); ++i) {
          samples_counter.push_back(sample_perf_counters[i]);
        }

        for (unsigned i = 0; i < (num_pwr_cmps); ++i) {
          pwr_counter.push_back(sample_cmp_pwr[i]);
        }
        assert(pwr_counter.size() == (double)num_pwr_cmps);
        assert(samples_counter.size() == (double)num_perf_counters);
      } else {
        // Get current average
        double temp_avg = sample_val / (double)samples.size();

        if (abs(proc->rt_power.readOp.dynamic - temp_avg) <
            gpu_steady_power_deviation) {  // Value is within threshold
          sample_val += proc->rt_power.readOp.dynamic;
          samples.push_back(proc->rt_power.readOp.dynamic);
          for (unsigned i = 0; i < (num_perf_counters); ++i) {
            samples_counter.at(i) += sample_perf_counters[i];
          }

          for (unsigned i = 0; i < (num_pwr_cmps); ++i) {
            pwr_counter.at(i) += sample_cmp_pwr[i];
          }

        } else {  // Value exceeds threshold, not considered steady state
          print_steady_state(position, init_val);
        }
      }
    } else {
      print_steady_state(position, init_val);
    }
    gzclose(steady_state_tacking_file);
  }
}

void gpgpu_sim_wrapper::open_files() {
  if (g_power_simulation_enabled) {
    if (g_power_trace_enabled) {
      power_trace_file = gzopen(g_power_trace_filename, "a");
      metric_trace_file = gzopen(g_metric_trace_filename, "a");
    }
  }
}
void gpgpu_sim_wrapper::close_files() {
  if (g_power_simulation_enabled) {
    if (g_power_trace_enabled) {
      gzclose(power_trace_file);
      gzclose(metric_trace_file);
    }
  }
}


double gpgpu_sim_wrapper::per_core_power_calculation(double* Cluster_freq) {


}