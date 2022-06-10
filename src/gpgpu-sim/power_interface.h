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

#ifndef POWER_INTERFACE_H_
#define POWER_INTERFACE_H_

#include "gpu-sim.h"
#include "power_stat.h"
#include "shader.h"

#include "gpgpu_sim_wrapper.h"

void init_mcpat(const gpgpu_sim_config &config,
                class gpgpu_sim_wrapper *wrapper, unsigned stat_sample_freq,
                unsigned tot_inst, unsigned inst);
bool mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst,class simt_core_cluster **,int,float*,double*,float *,double &Total_exe_time,double *new_cluster_freq
                 );
void mcpat_reset_perf_count(class gpgpu_sim_wrapper *wrapper);
void mcpat_cycle_power_calculation(double &Power,double &Throughput,int &constraint,double &power_time,
                                     const class gpgpu_sim_wrapper *wrapper,
                                     double * cluster_freq_new,double* base_cluster_freq
                                   ,double Actual_power,double Total_Throughput);

#endif /* POWER_INTERFACE_H_ */












































































































