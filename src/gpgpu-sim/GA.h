//
// Created by pouria on 5/23/22.
//

#ifndef OPENGA_TEST_GA_H
#define OPENGA_TEST_GA_H

#include "gpu-sim.h"
#include "power_stat.h"
#include "shader.h"
#include "vector"
#include "gpgpu_sim_wrapper.h"


class Population{
 public:
  Population(unsigned ,unsigned,unsigned ,double *,float,float,float);
  class Chromosome** parent_chromosomes;
  class Chromosome** child_chromosomes;
  double *available_alleles;
  unsigned number_avai_alleles;
  unsigned population_size;
  unsigned chromosome_size;
  double sum_powers_cons_one;
  double sum_powers_cons_zero;
  double sum_powers_time;
  float H_crossover;
  float H_fittest;
  float H_mutation;
  double *base_cluster_freq;
//Mcpat data to be past to objective function
  const gpgpu_sim_config *config;
  const shader_core_config *shdr_config;
  class gpgpu_sim_wrapper *wrapper;
  class power_stat_t *power_stats;
  unsigned stat_sample_freq;
  unsigned tot_cycle;
  unsigned cycle;
  unsigned tot_inst;
  unsigned inst;
  double base_freq;
  class simt_core_cluster **m_cluster;
  int shaders_per_cluster;
  float* numb_active_sms;
  double * cluster_freq;
  float* num_idle_core_per_cluster;
  float *average_pipeline_duty_cycle_per_sm;
  double Power;
  std::vector<double> Throughput;
  double Total_Throughput;
//

  void population_init();
  unsigned parent_selection_SUS(std::vector<struct array_data>Array_con);
  void calculate_sum_of_powers(std::vector<struct array_data>Array_cons,double &sum_powers_cons);
  void calculate_cumulative_probability(std::vector<struct array_data>Array_cons,double &sum_powers_cons);
  void swap_mutation(unsigned parent_index,unsigned child_idx);
  void bit_flip_mutation(unsigned,unsigned );
  void one_point_crossover(unsigned first_parent,unsigned second_parent,unsigned child_idx);
  void All_chromosomes_Power();
  void Per_chromosomes_power(unsigned idx,unsigned );
  void mcpat_data_set(class gpgpu_sim_wrapper *wrapper,
                      double* base_cluster_freq,std::vector<double> Throughput,double Power);
  class Binary_tree* power_tree;
  double* evolution(unsigned number_iterations,double *,double Actual_power,double &new_Throughput,double &new_power);
  unsigned generate_random_values(unsigned min,unsigned max);
};

class Chromosome{
 public:
  Chromosome(unsigned);
  double *gene;
  unsigned size;
  double power;
  double Throughput;
  double power_time;
  int constraint;
  double power_calculation();
  float probability;
  float cumulative_probability;
  unsigned generate_random_values(unsigned min,unsigned max);
  void calculate_probability(double sum_power);

};

struct Node{
  unsigned value;
  unsigned idx;
  int constraint;
  struct Node* left;
  struct Node* right;
};

struct array_data{
  unsigned idx;
  unsigned value;
  int constraint;
};

class Binary_tree{
 public:
  struct Node* root;
  struct Node* create_node(double value,int constraint,unsigned idx);
  void create_tree(unsigned num_nodes,class Chromosome **);
  void insert(struct Node* root,double value,int constraint,unsigned idx);
  void create_arrays(struct Node* node,std::vector<struct array_data>&,
                     std::vector<struct array_data>&);

  void print_(std::vector<struct array_data>Array_con_one,
              std::vector<struct array_data>Array_con_zero);

  struct array_data find_min(std::vector<struct array_data>&Array_con_one,
                             std::vector<struct array_data>&Array_con_zero);
  void delete_min_node();
  unsigned num_nodes;
  unsigned num_nodes_with_cns_one;
  unsigned counter;

};

#endif //OPENGA_TEST_GA_H
