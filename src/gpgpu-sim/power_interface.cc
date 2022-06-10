// Copyright (c) 2009-2011, Tor M. Aamodt, Ahmed El-Shafiey, Tayler Hetherington
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

#include "power_interface.h"
#include "GA.h"
#include <fstream>
void init_mcpat(const gpgpu_sim_config &config,
                class gpgpu_sim_wrapper *wrapper, unsigned stat_sample_freq,
                unsigned tot_inst, unsigned inst) {
  wrapper->init_mcpat(
      config.g_power_config_name, config.g_power_filename,
      config.g_power_trace_filename, config.g_metric_trace_filename,
      config.g_steady_state_tracking_filename,
      config.g_power_simulation_enabled, config.g_power_trace_enabled,
      config.g_steady_power_levels_enabled, config.g_power_per_cycle_dump,
      config.gpu_steady_power_deviation, config.gpu_steady_min_period,
      config.g_power_trace_zlevel, tot_inst + inst, stat_sample_freq);
}

bool mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst,class simt_core_cluster **m_cluster,int shaders_per_cluster,\
                 float* numb_active_sms,double * cluster_freq,float* average_pipeline_duty_cycle_per_sm
                    ,double &Total_exe_time,double* new_cluster_freq) {
  static bool mcpat_init = true;

  if (mcpat_init) {  // If first cycle, don't have any power numbers yet
    mcpat_init = false;
    return 0;
  }
  for(int i=0;i<wrapper->number_shaders;i++)
    wrapper->p_cores[i]->sys.core[0].clock_rate = (int)(cluster_freq[i]/((1e6)));

  if ((tot_cycle + cycle) % stat_sample_freq == 0) {
    Total_exe_time += (stat_sample_freq/cluster_freq[0]);
    double *tot_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *total_int_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *tot_fp_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *tot_commited_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);



    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes, stat_sample_freq,
        stat_sample_freq, power_stats->get_total_inst(tot_ins_set_inst_power),
        power_stats->get_total_int_inst(total_int_ins_set_inst_power),
        power_stats->get_total_fp_inst(tot_fp_ins_set_inst_power),
        power_stats->get_l1d_read_accesses(),
        power_stats->get_l1d_write_accesses(),
        power_stats->get_committed_inst(tot_commited_ins_set_inst_power),
        tot_ins_set_inst_power, total_int_ins_set_inst_power,
        tot_fp_ins_set_inst_power, tot_commited_ins_set_inst_power,cluster_freq,stat_sample_freq);

    FILE * total_cycle_file;
    total_cycle_file = fopen("/home/pouria/Desktop/G_GPU/original_freq_per_sm/src/gpgpu-sim/total_cycle.txt","a");
    fprintf(total_cycle_file,"\n");
    for (int i = 0; i < wrapper->number_shaders; i++)
      fprintf(total_cycle_file,"%lf ",wrapper->p_cores[i]->sys.core[0].total_cycles);
    fclose(total_cycle_file);

    // Single RF for both int and fp ops
    double *regfile_reads_set_regfile_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *regfile_writes_set_regfile_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *non_regfile_operands_set_regfile_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);

    wrapper->set_regfile_power(
        power_stats->get_regfile_reads(regfile_reads_set_regfile_power),
        power_stats->get_regfile_writes(regfile_writes_set_regfile_power),
        power_stats->get_non_regfile_operands(
            non_regfile_operands_set_regfile_power),
        regfile_reads_set_regfile_power, regfile_writes_set_regfile_power,
        non_regfile_operands_set_regfile_power);

    // Instruction cache stats
    wrapper->set_icache_power(power_stats->get_inst_c_hits(),
                              power_stats->get_inst_c_misses());

    // Constant Cache, shared memory, texture cache
    wrapper->set_ccache_power(power_stats->get_constant_c_hits(),
                              power_stats->get_constant_c_misses());
    wrapper->set_tcache_power(power_stats->get_texture_c_hits(),
                              power_stats->get_texture_c_misses());

    double *shmem_read_set_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    wrapper->set_shrd_mem_power(
        power_stats->get_shmem_read_access(shmem_read_set_power),
        shmem_read_set_power);

    wrapper->set_l1cache_power(
        power_stats->get_l1d_read_hits(), power_stats->get_l1d_read_misses(),
        power_stats->get_l1d_write_hits(), power_stats->get_l1d_write_misses());

    wrapper->set_l2cache_power(
        power_stats->get_l2_read_hits(), power_stats->get_l2_read_misses(),
        power_stats->get_l2_write_hits(), power_stats->get_l2_write_misses());
    //    free()
    float active_sms = 0;
    FILE *file;
    char *string_ =
        "/home/pouria/Desktop/G_GPU/original_freq_per_sm/src/gpgpu-sim/activesms.txt";
    file = fopen(string_, "a");
    fprintf(file, "loop\n");
    float *active_sms_per_cluster =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float *num_cores_per_cluster =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float *num_idle_core_per_cluster =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float total_active_sms = 0;
    for (int i = 0; i < wrapper->number_shaders; i++) {
      active_sms_per_cluster[i] = numb_active_sms[i] * ((cluster_freq[0]) /
                                  cluster_freq[i] )/ stat_sample_freq;
      active_sms += active_sms_per_cluster[i];
      num_cores_per_cluster[i] = shaders_per_cluster;
      num_idle_core_per_cluster[i] =
          num_cores_per_cluster[i] - active_sms_per_cluster[i];
      fprintf(file, "%d : %f %f %f\n", i, numb_active_sms[i],active_sms_per_cluster[i], num_idle_core_per_cluster[i]);
    }
    fclose(file);
    float num_cores = shdr_config->num_shader();
    float num_idle_core = num_cores - active_sms;

    wrapper->set_idle_core_power(num_idle_core, num_idle_core_per_cluster);

    // pipeline power - pipeline_duty_cycle *= percent_active_sms;
    FILE* file_DUTY;
    file_DUTY = fopen("/home/pouria/Desktop/G_GPU/DATA/duty.txt","a");

    float* pipeline_duty_cycle_per_sm = (float*)malloc(sizeof(float) *wrapper->number_shaders);
    for (int i = 0; i < wrapper->number_shaders; i++) {
      pipeline_duty_cycle_per_sm[i] = average_pipeline_duty_cycle_per_sm[i] * (cluster_freq[0]/cluster_freq[i]) / stat_sample_freq < 0.8\
          ?average_pipeline_duty_cycle_per_sm[i] * (cluster_freq[0]/cluster_freq[i]) / stat_sample_freq : 0.8;
      fprintf(file_DUTY,"\n%d: %f %f",i, pipeline_duty_cycle_per_sm[i],average_pipeline_duty_cycle_per_sm[i]);
    }
    fclose(file_DUTY);
    float pipeline_duty_cycle =
        ((*power_stats->m_average_pipeline_duty_cycle / (stat_sample_freq)) <
         0.8)
            ? ((*power_stats->m_average_pipeline_duty_cycle) / stat_sample_freq)
            : 0.8;
    wrapper->set_duty_cycle_power(pipeline_duty_cycle,pipeline_duty_cycle_per_sm);


    // Memory Controller
    wrapper->set_mem_ctrl_power(power_stats->get_dram_rd(),
                                power_stats->get_dram_wr(),
                                power_stats->get_dram_pre());

    // Execution pipeline accesses
    // FPU (SP) accesses, Integer ALU (not present in Tesla), Sfu accesses
    double *tot_fpu_accessess_set_exec_unit_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *ialu_accessess_set_exec_unit_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *tot_sfu_accessess_set_exec_unit_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);

    wrapper->set_exec_unit_power(
        power_stats->get_tot_fpu_accessess(
            tot_fpu_accessess_set_exec_unit_power),
        power_stats->get_ialu_accessess(ialu_accessess_set_exec_unit_power),
        power_stats->get_tot_sfu_accessess(
            tot_sfu_accessess_set_exec_unit_power),
        tot_fpu_accessess_set_exec_unit_power,
        ialu_accessess_set_exec_unit_power,
        tot_sfu_accessess_set_exec_unit_power);

    // Average active lanes for sp and sfu pipelines
    float *sp_active_lanes_set_active_lanes_power =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float *sfu_active_lanes_set_active_lanes_power =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float avg_sp_active_lanes = power_stats->get_sp_active_lanes(
        sp_active_lanes_set_active_lanes_power, cluster_freq, stat_sample_freq);
    float avg_sfu_active_lanes = (power_stats->get_sfu_active_lanes(
        sfu_active_lanes_set_active_lanes_power, cluster_freq,
        stat_sample_freq));
    assert(avg_sp_active_lanes <= 32);
    assert(avg_sfu_active_lanes <= 32);
    //    for(int i=0;i<wrapper->number_shaders;i++)
    //    {
    //      assert(sp_active_lanes_set_active_lanes_power[i]/ stat_sample_freq <= 32); assert(sfu_active_lanes_set_active_lanes_power[i]/ stat_sample_freq <= 32);
    //    }

    wrapper->set_active_lanes_power((power_stats->get_sp_active_lanes(
                                        sp_active_lanes_set_active_lanes_power,
                                        cluster_freq, stat_sample_freq)),
                                    (power_stats->get_sfu_active_lanes(
                                        sfu_active_lanes_set_active_lanes_power,
                                        cluster_freq, stat_sample_freq)),
                                    sp_active_lanes_set_active_lanes_power,
                                    sfu_active_lanes_set_active_lanes_power,
                                    stat_sample_freq);

    double *n_icnt_mem_to_simt_set_NoC_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *n_icnt_simt_to_mem_set_NoC_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);

    double n_icnt_simt_to_mem =
        (double)
            power_stats->get_icnt_simt_to_mem(n_icnt_simt_to_mem_set_NoC_power);  // # flits from SIMT clusters
                                                                                  // to memory partitions
    double n_icnt_mem_to_simt =
        (double)
            power_stats->get_icnt_mem_to_simt(n_icnt_mem_to_simt_set_NoC_power);  // # flits from memory
                                                                                  // partitions to SIMT clusters
    wrapper->set_NoC_power(
        n_icnt_mem_to_simt,
        n_icnt_simt_to_mem,n_icnt_mem_to_simt_set_NoC_power,n_icnt_simt_to_mem_set_NoC_power);  // Number of flits traversing the interconnect

    FILE* IBP_ = fopen("/home/pouria/Desktop/G_GPU/DATA/IBP.txt","a");
    FILE * file_final_ifu;
    file_final_ifu = fopen("/home/pouria/Desktop/G_GPU/DATA/CORE_PIP_ifu.txt","a");
    for(int i=0;i<wrapper->number_shaders;i++) {
      fprintf(file_final_ifu, "\n%d: %lf", i, cluster_freq[i]);
      fprintf(IBP_, "\n%d: %lf", i, cluster_freq[i]);
    }
    fflush(file_final_ifu);
    fclose(file_final_ifu);
    fclose(IBP_);

    wrapper->compute(true);

    wrapper->update_components_power(1);
    wrapper->update_components_power_per_core(0,cluster_freq[0]);
    double Actual_power = wrapper->sum_per_sm_and_shard_power(cluster_freq);
//    FILE * file_final;
//    file_final = fopen("/home/pouria/Desktop/G_GPU/DATA/FINAL.txt","a");
//    fprintf(file_final,"\n%lf %lf",cluster_freq[0],Actual_power);
//    fflush(file_final);
//    fclose(file_final);

    wrapper->print_trace_files();


    wrapper->detect_print_steady_state(0, tot_inst + inst);

    wrapper->power_metrics_calculations();
    wrapper->smp_cpm_pwr_print();
    wrapper->dump();
    FILE * exetime;
    exetime = fopen("/home/pouria/Desktop/G_GPU/DATA/Exetime.txt","a");

      fprintf(exetime, "%1.12lf: %5.12lf ",
              wrapper->proc_cores[0]->cores[0]->executionTime,Total_exe_time);

    fprintf(exetime,"\n");
    fclose(exetime);

    double * Available_freqs = (double *)malloc(sizeof(double )*10);
    for(int i=0;i<10;i++)
      Available_freqs[i] = (i+1)*100*1e6;
    class Population pop = Population(10,wrapper->number_shaders,10,Available_freqs,0.2,0.2,0.4);
    pop.mcpat_data_set(wrapper, cluster_freq,wrapper->Throughput,Actual_power);
    double new_Throughput;
    double new_power;
    double original_throughput=0;
    double* optimized_freq = pop.evolution(10,cluster_freq,Actual_power,new_Throughput,new_power);

    FILE * Final_freqs;
    Final_freqs = fopen("/home/pouria/Desktop/G_GPU/DATA/Final_freqs.txt","a");
    fprintf(Final_freqs,"\n**********\n");
    for(int i=0;i<wrapper->number_shaders;i++) {
      original_throughput += wrapper->Throughput[i];
      if(i%5==0)
        fprintf(Final_freqs,"\n");
      fprintf(Final_freqs,"%d: %lf %1.12lf\t",i,optimized_freq[i],1/optimized_freq[i]);
  //      if(i==0)
  //        new_cluster_freq[i] = 700*1e6;
  //      else
        new_cluster_freq[i] = optimized_freq[i];
    }
    fclose(Final_freqs);

    power_stats->save_stats(wrapper->number_shaders, numb_active_sms,average_pipeline_duty_cycle_per_sm);
    double performance_loss = (original_throughput-new_Throughput)/original_throughput;
    double Power_gain = (Actual_power - new_power)/Actual_power;

    std::ofstream file_pre;
    file_pre.open("/home/pouria/Desktop/G_GPU/DATA/prediction.txt",std::ios::app);
    file_pre << "\n************\n";
    file_pre<< "Original power: "<< Actual_power <<"\ncalculated power: " <<new_power
        <<"\noriginal Throughput: "<< original_throughput << "\nCalculated Throughput"<< new_Throughput
         <<"Perfromance loss: "<<performance_loss<<std::endl;
    file_pre.close();

    if( Power_gain > 1.5*performance_loss &&  performance_loss<0.1 && new_cluster_freq[0] >500*1e6 )
      return 1;
    else
      return 0;
  }
  else
    return 0;
}

void mcpat_cycle_power_calculation( double &Power,double &Throughput,int &constraint,double &power_time,
                                   const class gpgpu_sim_wrapper *wrapper,
                                   double * cluster_freq_new,double* base_cluster_freq,double Actual_power,double Total_Throughput){
  double Total_power = 0;
  Throughput = 0;
  double alpha;
  double AVG_alpha_old = 0;
  double AVG_alpha_new = 0;
  double Power_gain;
  double Performance_loss;
  for(int i=0;i<wrapper->number_shaders;i++){
    alpha = (cluster_freq_new[i]/base_cluster_freq[i]);
    Throughput += wrapper->Throughput[i] * alpha;
    Total_power += wrapper->sample_cmp_pwr_S[i]*alpha;
    AVG_alpha_new += cluster_freq_new[i];
    AVG_alpha_old += base_cluster_freq[i];
  }

  AVG_alpha_new /= wrapper->number_shaders;
  AVG_alpha_old /= wrapper->number_shaders;

  Total_power += wrapper->sample_cmp_pwr_Shrd * (AVG_alpha_new/AVG_alpha_old);
  Total_power += wrapper->sample_cmp_pwr_const;
  Power = Total_power;
  if(Throughput == 0)
    Throughput = 1;
  power_time =  Power;
  Power_gain = (Actual_power - Power)/Actual_power;
  Performance_loss = (Total_Throughput - Throughput )/Total_Throughput;
  constraint = 1;
  if(Power_gain<=0)
    constraint = 0;
  if(Performance_loss>0.1 || cluster_freq_new[0] <600*1e6)
    constraint = 0;
}




//double mcpat_cycle_power_calculation(const gpgpu_sim_config &config,
//                                   const shader_core_config *shdr_config,
//                                   class gpgpu_sim_wrapper *wrapper,
//                                   class power_stat_t *power_stats, unsigned stat_sample_freq,
//                                   unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
//                                   unsigned inst,class simt_core_cluster **m_cluster,int shaders_per_cluster,\
//                                   float* numb_active_sms,double * cluster_freq,float* num_idle_core_per_cluster,float* average_pipeline_duty_cycle_per_sm) {
//  double Calculated_power;
//  (wrapper->return_p())->sys.total_cycles = stat_sample_freq;
//  FILE* file;
//  FILE *exetime;
//  file = fopen("/home/pouria/Desktop/G_GPU/original_freq_per_sm/src/gpgpu-sim/sample_stat_freq.txt","a");
//  fprintf(file,"\nstat_sample_freq: %u",stat_sample_freq);
//
//  for(int i=0;i<wrapper->number_shaders;i++) {
//    wrapper->p_cores[i]->sys.core[0].clock_rate = (int)(cluster_freq[i]/((1e6)));
//    wrapper->p_cores[i]->sys.total_cycles =
//        stat_sample_freq * cluster_freq[i] / cluster_freq[0];
//    fprintf(file,"\ntotal_cycles %d: %lf",i,wrapper->p_cores[i]->sys.total_cycles);
//  }
//
//  fclose(file);
//  double *tot_ins_set_inst_power =
//      (double *)malloc(sizeof(double) * wrapper->number_shaders);
//  double *total_int_ins_set_inst_power =
//      (double *)malloc(sizeof(double) * wrapper->number_shaders);
//  double *tot_fp_ins_set_inst_power =
//      (double *)malloc(sizeof(double) * wrapper->number_shaders);
//  double *tot_commited_ins_set_inst_power =
//      (double *)malloc(sizeof(double) * wrapper->number_shaders);
//
//
//  wrapper->set_inst_power(
//      shdr_config->gpgpu_clock_gated_lanes, stat_sample_freq,
//      stat_sample_freq, power_stats->get_total_inst(tot_ins_set_inst_power),
//      power_stats->get_total_int_inst(total_int_ins_set_inst_power),
//      power_stats->get_total_fp_inst(tot_fp_ins_set_inst_power),
//      power_stats->get_l1d_read_accesses(),
//      power_stats->get_l1d_write_accesses(),
//      power_stats->get_committed_inst(tot_commited_ins_set_inst_power),
//      tot_ins_set_inst_power, total_int_ins_set_inst_power,
//      tot_fp_ins_set_inst_power, tot_commited_ins_set_inst_power,cluster_freq);
//  file = fopen("/home/pouria/Desktop/G_GPU/original_freq_per_sm/src/gpgpu-sim/data.txt","a");
//  fprintf(file,"\nclock_lanes total_cycles busy_cycles total_inst int_inst fp_inst load_inst",stat_sample_freq);
//  fprintf(file,"\n%i %lf %lf %lf %lf %lf %lf %lf", wrapper->p_cores[0]->sys.core[0].gpgpu_clock_gated_lanes,\
//          wrapper->p_cores[0]->sys.core[0].total_cycles,wrapper->p_cores[0]->sys.core[0].busy_cycles,\
//          wrapper->p_cores[0]->sys.core[0].total_instructions,wrapper->p_cores[0]->sys.core[0].int_instructions,\
//          wrapper->p_cores[0]->sys.core[0].fp_instructions,wrapper->p_cores[0]->sys.core[0].load_instructions);
//  fclose(file);
//
//
//  wrapper->compute(false);
//
//  FILE* IBP_ = fopen("/home/pouria/Desktop/G_GPU/DATA/IBP.txt","a");
//  for(int i=0;i<wrapper->number_shaders;i++) {
//    fprintf(IBP_, "\n%d: %lf", i, cluster_freq[i]);
//  }
//  fclose(IBP_);
//  //      Calculated_power = wrapper->update_components_power(0);
//
//
//  wrapper->update_components_power(0);
//  wrapper->update_components_power_per_core(1,cluster_freq[0]);
//  Calculated_power = wrapper->sum_per_sm_and_shard_power(cluster_freq);
//  exetime = fopen("/home/pouria/Desktop/G_GPU/DATA/Exetime.txt","a");
//  fprintf(exetime,"\n     ***genetic functions***\n");
//  for(int i=0;i<wrapper->number_shaders;i++) {
//    if(!(i%5))
//      fprintf(exetime,"\n");
//    fprintf(exetime, "[%d: %2.10lf] ",i,
//            wrapper->proc_cores[i]->cores[0]->executionTime);
//  }
//  fprintf(exetime,"\n");
//  fclose(exetime);
//
//  FILE * file_final;
//  file_final = fopen("/home/pouria/Desktop/G_GPU/DATA/FINAL_cycle.txt","a");
//  fprintf(file_final,"\n%lf %lf",cluster_freq[0],Calculated_power);
//  fflush(file_final);
//  fclose(file_final);
//
//  free(tot_ins_set_inst_power);
//  free(total_int_ins_set_inst_power);
//  free(tot_fp_ins_set_inst_power);
//  free(tot_commited_ins_set_inst_power);
//  return Calculated_power;
//}

//    double *Cluster_freq =
//        (double *)malloc(sizeof(double) * wrapper->number_shaders);
//
////    int* clock_rate = (int *)malloc(sizeof(int) * wrapper->number_shaders);
////    double* total_cycles = (double *)malloc(sizeof(double) * wrapper->number_shaders);
////    double* core_total_cycles = (double *)malloc(sizeof(double) * wrapper->number_shaders);
////    double* busy_cycles = (double *)malloc(sizeof(double) * wrapper->number_shaders);
////    double* sp_average_active_lanes = (double *)malloc(sizeof(double) * wrapper->number_shaders);
////    double* sfu_average_active_lanes = (double *)malloc(sizeof(double) * wrapper->number_shaders);
////
////    for(int i=0;i<wrapper->number_shaders;i++){
////      clock_rate[i] =  wrapper->p_cores[i]->sys.core[0].clock_rate;
////      total_cycles[i] = wrapper->p_cores[i]->sys.total_cycles;
////      core_total_cycles[i] =  wrapper->p_cores[i]->sys.core[0].total_cycles;
////      busy_cycles[i] =  wrapper->p_cores[i]->sys.core[0].busy_cycles;
////      sp_average_active_lanes[i] = wrapper->p_cores[i]->sys.core[0].sp_average_active_lanes;
////      sfu_average_active_lanes[i] =  wrapper->p_cores[i]->sys.core[0].sfu_average_active_lanes;
////      printf("\nNormal %d clock rate: %d\ttotal cycle %lf\tcoretotal cycle %lf\nbusy cycle %lf sp %lf sfu %lf",i,clock_rate[i],\
////             total_cycles[i], core_total_cycles[i],busy_cycles[i],sp_average_active_lanes[i], sfu_average_active_lanes[i]);
////    }
//printf("\nLoop:");
//    for (int k = 0; k < 3; k++) {
//      for (int i = 0; i < wrapper->number_shaders; i++) {
//        Cluster_freq[i] = 100 * (1 + k / 5) * (i % 3 + 1) * (1 << 20);
//
//        wrapper->p_cores[i]->sys.core[0].clock_rate =
//            (int)(Cluster_freq[i] / ((1 << 20)));
////            printf("\ncluster fre: %lf Cluster freq: %lf",cluster_freq[0],Cluster_freq[i]);
////        wrapper->p_cores[i]->sys.total_cycles = total_cycles[i] * Cluster_freq[i] /
////            cluster_freq[i];
////        wrapper->p_cores[i]->sys.core[0].total_cycles =
////            core_total_cycles[i] * Cluster_freq[i] / cluster_freq[i];
////        wrapper->p_cores[i]->sys.core[0].busy_cycles =
////            busy_cycles[i] * Cluster_freq[i] /
////            cluster_freq[i];
////        wrapper->p_cores[i]->sys.core[0].sp_average_active_lanes =
////            sp_average_active_lanes[i] *
////            Cluster_freq[i] / cluster_freq[i];
////        wrapper->p_cores[i]->sys.core[0].sfu_average_active_lanes =
////            sfu_average_active_lanes[i]*
////            Cluster_freq[i] / cluster_freq[i];
////        printf("\n%d clock rate: %d\ttotal cycle %lf\tcoretotal cycle %lf\nbusy cycle %lf sp %lf sfu %lf",i, wrapper->p_cores[i]->sys.core[0].clock_rate,\
////               wrapper->p_cores[i]->sys.total_cycles, wrapper->p_cores[i]->sys.core[0].total_cycles, wrapper->p_cores[i]->sys.core[0].busy_cycles\
////               , wrapper->p_cores[i]->sys.core[0].sp_average_active_lanes, wrapper->p_cores[i]->sys.core[0].sfu_average_active_lanes);
////        printf("/nchanges");
////        printf("")
//      }
//
//      wrapper->compute();
//
//        wrapper->update_components_power(0);
//      wrapper->update_components_power_per_core();
//      wrapper->print_trace_files();
//      power_stats->save_stats(wrapper->number_shaders, numb_active_sms);
//
//      wrapper->detect_print_steady_state(0, tot_inst + inst);
//
//      wrapper->power_metrics_calculations();
//      wrapper->smp_cpm_pwr_print();


//    free(clock_rate);
//    free(total_cycles);
//    free(core_total_cycles);
//    free(busy_cycles);
//    free(sp_average_active_lanes);
//    free(sfu_average_active_lanes);

  // wrapper->close_files();
//double simple_power_prediction(class gpgpu_sim_wrapper *wrapper,double* base_freq,double *cluster_freq,unsigned num_clusters){
//  double IBP_power        = 0 ;
//  double SHRDP_power      = 0 ;
//  double RFP_power        = 0 ;
//  double SPP_power        = 0 ;
//  double SFUP_power       = 0 ;
//  double FPUP_power       = 0 ;
//  double SCHEDP_power     = 0 ;
//  double NOCP_power       = 0 ;
//  double PIPEP_power      = 0 ;
//  double IDLE_COREP_power = 0 ;
//  double ICP_power   = 0;
//  double DCP_power   = 0;
//  double TCP_power   = 0;
//  double CCP_power   = 0;
//  double L2CP_power  = 0;
//  double MCP_power   = 0;
//  double DRAMP_power = 0;
//  double AVG_Freq_new = 0;
//  double AVG_Freq_base = 0;
//  double CONST_DYNAMICP_power = wrapper->sample_cmp_pwr[CONST_DYNAMICP];
//  for(int i=0;i<num_clusters;i++) {
//    IBP_power +=       (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][IBP];
//    SHRDP_power +=     (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][SHRDP];
//    RFP_power +=       (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][RFP];
//    SPP_power +=       (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][SPP];
//    SFUP_power +=      (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][SFUP];
//    FPUP_power +=      (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][FPUP];
//    SCHEDP_power +     (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][SCHEDP];
//    NOCP_power +=      (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][NOCP] ;
//    PIPEP_power +=     (cluster_freq[i] / base_freq[i]) * wrapper->sample_cmp_pwr[i][PIPEP];
//    IDLE_COREP_power+= (cluster_freq[i] / base_freq[i]  * wrapper->sample_cmp_pwr[i][IDLE_COREP];
//
//    AVG_Freq_new += cluster_freq[i];
//    AVG_Freq_base += cluster_freq[i];
//  }
//  AVG_Freq_new /= num_clusters;
//  AVG_Freq_base /= num_clusters;
//
//  ICP_power   = (AVG_Freq_new / AVG_Freq_base) * wrapper->sample_cmp_pwr[ICP]
//  DCP_power   = (AVG_Freq_new / AVG_Freq_base) * wrapper->sample_cmp_pwr[DCP]
//  TCP_power   = (AVG_Freq_new / AVG_Freq_base) * wrapper->sample_cmp_pwr[TCP]
//  CCP_power   = (AVG_Freq_new / AVG_Freq_base) * wrapper->sample_cmp_pwr[CCP]
//  L2CP_power  = (AVG_Freq_new / AVG_Freq_base) * wrapper->sample_cmp_pwr[L2CP]
//  MCP_power   = (AVG_Freq_new / AVG_Freq_base) * wrapper->sample_cmp_pwr[MCP]
//  DRAMP_power = (AVG_Freq_new / AVG_Freq_base) * wrapper->sample_cmp_pwr[DRAMP]
//
//   // This constant dynamic power (e.g., clock power) part is estimated via
//  // regression model.
//  sample_cmp_pwr[CONST_DYNAMICP] = 0;
//  double cnst_dyn =
//      proc->get_const_dynamic_power() / (proc->cores[0]->executionTime);
//  // If the regression scaling term is greater than the recorded constant
//  // dynamic power then use the difference (other portion already added to
//  // dynamic power). Else, all the constant dynamic power is accounted for, add
//  // nothing.
//  if (p->sys.scaling_coefficients[CONST_DYNAMICN] > cnst_dyn)
//    sample_cmp_pwr[CONST_DYNAMICP] =
//        (p->sys.scaling_coefficients[CONST_DYNAMICN] - cnst_dyn);
//}

void mcpat_reset_perf_count(class gpgpu_sim_wrapper *wrapper) {
  wrapper->reset_counters();
}












