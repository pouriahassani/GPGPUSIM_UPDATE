//
// Created by pouria on 5/23/22.
//
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include "GA.h"
#include "power_interface.h"
#include "vector"

Population::Population(unsigned population_size,unsigned chromosome_size,unsigned number_avai_alleles,\
                       double* available_alleles,float H_crossover,float H_fittest,float H_mutation){
  this->population_size = population_size;
  this->chromosome_size = chromosome_size;
  this->available_alleles = available_alleles;
  this->number_avai_alleles = number_avai_alleles;
  this->H_crossover = H_crossover;
  this->H_fittest = H_fittest;
  this->H_mutation = H_mutation;

  parent_chromosomes = (class Chromosome**)malloc(sizeof(class Chromosome*)*population_size);
  child_chromosomes = (class Chromosome**)malloc(sizeof(class Chromosome*)*population_size);
  power_tree = (class Binary_tree*) malloc(sizeof(class Binary_tree));
  for(unsigned i=0;i<population_size;i++) {
    parent_chromosomes[i] = (class Chromosome*)malloc(sizeof(class Chromosome)*population_size);
    child_chromosomes[i] = (class Chromosome*)malloc(sizeof(class Chromosome)*population_size);
    *(parent_chromosomes[i]) = Chromosome(chromosome_size);
    *(child_chromosomes[i]) = Chromosome(chromosome_size);
  }

}

void Population::mcpat_data_set(class gpgpu_sim_wrapper *wrapper,
                                double* base_cluster_freq,std::vector<double> Throughput,double Power){

  this->wrapper = wrapper;
  this->base_cluster_freq = base_cluster_freq;
  this->Throughput = Throughput;
  this->Power = Power;
  for(int i=0;i<wrapper->number_shaders;i++)
    Total_Throughput = std::accumulate(Throughput.begin(),Throughput.end(),0);
}

void Chromosome::calculate_probability(double power_sum_time){
  probability = power_time/power_sum_time;
}

void Population::calculate_sum_of_powers(std::vector<struct array_data>Array_cons,double &sum_powers_cons){
  sum_powers_time = 0;
  for(unsigned i=0;i<Array_cons.size();i++)
    sum_powers_time += Array_cons[i].value;
}

void Population:: calculate_cumulative_probability(std::vector<struct array_data>Array_cons,double &sum_powers_cons){
  float sum = 0;
  calculate_sum_of_powers(Array_cons,sum_powers_cons);
  unsigned idx;
  for(unsigned i=0;i<Array_cons.size();i++){
    idx = Array_cons[i].idx;
    parent_chromosomes[idx]->calculate_probability(sum_powers_time);
    sum+=parent_chromosomes[idx]->probability;
    parent_chromosomes[idx]->cumulative_probability = sum;
  }
}

unsigned Population::parent_selection_SUS(std::vector<struct array_data>Array_con){
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(0, RAND_MAX); // define the range
  float r = (float)(distr(gen))/RAND_MAX;
  unsigned index=0;

  while(index<Array_con.size()-1 && parent_chromosomes[Array_con[index].idx]->cumulative_probability<r){
    index++;
  }

  return Array_con[index].idx;
}

void Population::swap_mutation(unsigned parent_index,unsigned child_idx){

  for(int i=0;i<child_chromosomes[0]->size;i++)
    child_chromosomes[child_idx]->gene[i] = parent_chromosomes[parent_index]->gene[i];

  unsigned first_gene = generate_random_values(0, parent_chromosomes[parent_index]->size-1);
  unsigned second_gene = generate_random_values(0, parent_chromosomes[parent_index]->size-1);;
  child_chromosomes[child_idx]->gene[first_gene] = parent_chromosomes[parent_index]->gene[second_gene];
  child_chromosomes[child_idx]->gene[second_gene] = parent_chromosomes[parent_index]->gene[first_gene];
}

void Population::bit_flip_mutation(unsigned parent_index,unsigned child_idx){
  for(int i=0;i<child_chromosomes[0]->size;i++)
    child_chromosomes[child_idx]->gene[i] = parent_chromosomes[parent_index]->gene[i];
  unsigned gene_idx = generate_random_values(0, parent_chromosomes[0]->size-1);
  unsigned freq_idx = generate_random_values(0,number_avai_alleles-1);
  child_chromosomes[child_idx]->gene[gene_idx] = available_alleles[freq_idx];
}

void Population::one_point_crossover(unsigned first_parent,unsigned second_parent,unsigned child_idx) {
  unsigned point = generate_random_values(0, child_chromosomes[child_idx]->size - 1);
  for (unsigned i = 0; i < child_chromosomes[child_idx]->size; i++) {
    if (i <= point) {
      child_chromosomes[child_idx]->gene[i] = parent_chromosomes[first_parent]->gene[i];
      child_chromosomes[child_idx + 1]->gene[i] = parent_chromosomes[second_parent]->gene[i];
    } else {
      child_chromosomes[child_idx]->gene[i] = parent_chromosomes[second_parent]->gene[i];
      child_chromosomes[child_idx + 1]->gene[i] = parent_chromosomes[first_parent]->gene[i];
    }
  }
}

unsigned Population::generate_random_values(unsigned min, unsigned max) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(min, max); // define the range
  return (unsigned)(distr(gen));
}

void Population::All_chromosomes_Power() {
//  for (unsigned i = 1; i < population_size; i++) {
////    std::cout << "parent_chromosomes: " << i <<std::endl;
//    for (unsigned j = 0; j < parent_chromosomes[i]->size; j++)
////      std::cout << " gene: " << j << " value: " <<parent_chromosomes[i]->gene[j];
//  }
  for(int i=1;i<population_size;i++){
    mcpat_cycle_power_calculation(parent_chromosomes[i]->power,
                                  parent_chromosomes[i]->Throughput,
                                  parent_chromosomes[i]->constraint,
                                  parent_chromosomes[i]->power_time,
                                  wrapper,
                                  parent_chromosomes[i]->gene,
                                  base_cluster_freq,
                                  Power, Total_Throughput);

//    std::cout<<"\nParent power "<<i<<": "<<parent_chromosomes[i]->power;
//    std::cout<<"\t"<<parent_chromosomes[i]->Throughput<<"\t"<<parent_chromosomes[i]->power_time<<std::endl;

  }
}

void Population::Per_chromosomes_power(unsigned idx,unsigned p){
  if(p) {
    mcpat_cycle_power_calculation(
        parent_chromosomes[idx]->power, parent_chromosomes[idx]->Throughput,parent_chromosomes[idx]->constraint,parent_chromosomes[idx]->power_time,
        wrapper, parent_chromosomes[idx]->gene, base_cluster_freq,Power,Total_Throughput);
  }
  else {
    mcpat_cycle_power_calculation(
        child_chromosomes[idx]->power, child_chromosomes[idx]->Throughput,child_chromosomes[idx]->constraint,child_chromosomes[idx]->power_time,
        wrapper, child_chromosomes[idx]->gene, base_cluster_freq,Power,Total_Throughput);
//    std::cout<<"child_chromosomes[idx]->power_time: " << child_chromosomes[idx]->power_time<<std::endl;
  }
}

void Population::population_init(){
  std::ofstream file_step;
  unsigned idx;
  file_step.open("/home/pouria/Desktop/G_GPU/DATA/file_step.txt",std::ios::app);
  file_step<<"\n****************************\n*********actual value*********\n";

  parent_chromosomes[0]->Throughput = 0;
      for (unsigned j = 0; j < parent_chromosomes[0]->size; j++) {
    parent_chromosomes[0]->gene[j] = base_cluster_freq[j];
    parent_chromosomes[0]->Throughput += Throughput[j];
  }
  parent_chromosomes[0]->power = Power;
  parent_chromosomes[0]->power_time = Power;
  parent_chromosomes[0]->constraint = 1;

  for (unsigned i = 1; i < population_size; i++) {
    for (unsigned j = 0; j < parent_chromosomes[i]->size; j++) {
      idx = generate_random_values(0, number_avai_alleles - 1);
      parent_chromosomes[i]->gene[j] = available_alleles[idx];
//      std::cout << "parent_chromosomes: " << i << " gene: " << j << " value: " <<parent_chromosomes[i]->gene[j]<<std::endl;
    }
  }
    //        calculate power value for all the chromosomes
  All_chromosomes_Power();
  for(int i=0;i<population_size;i++){
      file_step<<parent_chromosomes[i]->power<<"\t: "<<parent_chromosomes[i]->Throughput<<"\t: "<<parent_chromosomes[i]->power_time<<std::endl;
  }
  file_step.close();
}

double* Population::evolution(unsigned number_iterations,double* Actual_freq,double Actual_power,double &new_Throughput,double &new_power) {
  //    create first generation with random values
  population_init();
  unsigned idx;
  unsigned idy;
  unsigned child_chromosome_idx;
  double temp;
  unsigned muted_chromosome;
  unsigned second_parent;
  unsigned first_parent;
  class Chromosome** temp_chromosome;
  std::ofstream file;
  std::ofstream file_step;
  file.open("/home/pouria/Desktop/G_GPU/DATA/GA_output_power.txt",std::ios::app);
  file_step.open("/home/pouria/Desktop/G_GPU/DATA/file_step.txt",std::ios::app);
  for(unsigned itr=0;itr<number_iterations;itr++) {
    child_chromosome_idx = 0;

    //    create binary tree for finding min in the power values
    power_tree->create_tree(population_size, parent_chromosomes);
    std::vector<struct array_data> Array_con_one;
    std::vector<struct array_data> Array_con_zero;

    power_tree->create_arrays(power_tree->root,Array_con_one,Array_con_zero);
    calculate_cumulative_probability(Array_con_one,sum_powers_cons_one);
    calculate_cumulative_probability(Array_con_zero,sum_powers_cons_zero);

    power_tree->print_(Array_con_one,Array_con_zero);

    std::vector<struct array_data> Array_con_one_new;
    std::vector<struct array_data> Array_con_zero_new;
    Array_con_one_new.assign(Array_con_one.begin(), Array_con_one.end());
    Array_con_zero_new.assign(Array_con_zero.begin(), Array_con_zero.end());


    //    find top 10% best fits
    file_step<<"\n****************** "<<std::endl;
    for (int k = 0; k < (int)(population_size * H_fittest); k++) {
      idx = (power_tree->find_min(Array_con_one,Array_con_zero)).idx;
      file_step<<"\nfittest: "<<k<<"\tidx: "<<idx<<"\tpower: "<<parent_chromosomes[idx]->power<<"\tThrouput: "<<
          parent_chromosomes[idx]->Throughput<<"\tconstraint: "<<parent_chromosomes[idx]->constraint<<std::endl;
      child_chromosome_idx++;
      for (int i = 0; i < child_chromosomes[idx]->size; i++) {
        child_chromosomes[k]->gene[i] = parent_chromosomes[idx]->gene[i];
        if(i%5 == 0)
          file_step<<"\n";
        file_step <<"gene["<<i<<"]: "<< child_chromosomes[k]->gene[i]<<"\t";
      }
      child_chromosomes[k]->power = parent_chromosomes[idx]->power;
      child_chromosomes[k]->power_time = parent_chromosomes[idx]->power_time;
      child_chromosomes[k]->Throughput = parent_chromosomes[idx]->Throughput;
      child_chromosomes[k]->constraint = parent_chromosomes[idx]->constraint;
    }
//
    ////    swap mutation
    file_step <<"\nswap mutation"<<std::endl;
    for (int k = 0; k < (int) (population_size * H_mutation/2); k++) {
      if(Array_con_one_new.size() && Array_con_zero_new.size()){
      if(rand()%3)
        muted_chromosome = parent_selection_SUS(Array_con_one_new);
      else
        muted_chromosome = parent_selection_SUS(Array_con_zero_new);
      }
      else{
        if(Array_con_one_new.size())
          muted_chromosome = parent_selection_SUS(Array_con_one_new);
        else
          muted_chromosome = parent_selection_SUS(Array_con_zero_new);
      }

      file_step <<"\nparent["<<muted_chromosome<<"] power: "<< parent_chromosomes[muted_chromosome]->power
                <<"\t"<<parent_chromosomes[muted_chromosome]->Throughput<<"\t"<<parent_chromosomes[muted_chromosome]->constraint
                <<std::endl;
      swap_mutation(muted_chromosome, child_chromosome_idx);
      Per_chromosomes_power(child_chromosome_idx,0);
      file_step <<"\nchild["<<child_chromosome_idx<<"] power: "<< child_chromosomes[child_chromosome_idx]->power
                <<"\t"<<child_chromosomes[child_chromosome_idx]->Throughput<<"\t"<<child_chromosomes[child_chromosome_idx]->constraint
                <<std::endl;
      child_chromosome_idx++;
    }
//
//
    for (int k = 0; k < (int) (population_size * H_mutation/2); k++) {
      if(Array_con_zero_new.size())
        muted_chromosome = parent_selection_SUS(Array_con_zero_new);
      else
        muted_chromosome = parent_selection_SUS(Array_con_one_new);
      bit_flip_mutation(muted_chromosome, child_chromosome_idx);
      Per_chromosomes_power(child_chromosome_idx,0);
      file_step <<"\nparent["<<muted_chromosome<<"] power: "<< parent_chromosomes[muted_chromosome]->power
                <<"\t"<<parent_chromosomes[muted_chromosome]->Throughput<<"\t"<<parent_chromosomes[muted_chromosome]->constraint
                <<std::endl;
      file_step <<"\nChild["<<child_chromosome_idx<<"] power: "<< child_chromosomes[child_chromosome_idx]->power
                <<"\t"<<child_chromosomes[child_chromosome_idx]->Throughput<<"\t"<<child_chromosomes[child_chromosome_idx]->constraint
                <<std::endl;
      child_chromosome_idx++;
    }
    file_step <<"\n****************************\n";
//    file_step.close();
//
    //    CrossOver
    for (int k = 0; k < (int) (population_size * H_crossover); k++) {
      if(Array_con_one_new.size() && Array_con_zero_new.size()){
      first_parent = parent_selection_SUS(Array_con_zero_new);
      second_parent = parent_selection_SUS(Array_con_one_new);
      }
      else{
        if(Array_con_one_new.size()){
          first_parent = parent_selection_SUS(Array_con_one_new);
          second_parent = parent_selection_SUS(Array_con_one_new);
        }
        else{
          first_parent = parent_selection_SUS(Array_con_zero_new);
          second_parent = parent_selection_SUS(Array_con_zero_new);
        }

      }

      file_step <<"\nparent["<<first_parent<<"] power: "<< parent_chromosomes[first_parent]->power
                <<"\t"<<parent_chromosomes[first_parent]->Throughput<<"\t"<<parent_chromosomes[first_parent]->constraint
                <<std::endl;
      file_step <<"\nparent["<<second_parent<<"] power: "<< parent_chromosomes[second_parent]->power
                <<"\t"<<parent_chromosomes[second_parent]->Throughput<<"\t"<<parent_chromosomes[second_parent]->constraint
                <<std::endl;

      one_point_crossover(first_parent, second_parent,child_chromosome_idx);
      Per_chromosomes_power(child_chromosome_idx,0);
      Per_chromosomes_power(child_chromosome_idx+1,0);
      file_step <<"\nchild["<<child_chromosome_idx<<"] power: "<< child_chromosomes[child_chromosome_idx]->power
                <<"\t"<<child_chromosomes[child_chromosome_idx]->Throughput<<"\t"<<child_chromosomes[child_chromosome_idx]->constraint
                <<std::endl;
      file_step <<"\nchild["<<child_chromosome_idx+1<<"] power: "<< child_chromosomes[child_chromosome_idx+1]->power
                <<"\t"<<child_chromosomes[child_chromosome_idx+1]->Throughput<<"\t"<<child_chromosomes[child_chromosome_idx+1]->constraint
                <<std::endl;
      child_chromosome_idx+=2;
    }

//
    temp_chromosome = child_chromosomes;
    child_chromosomes = parent_chromosomes;
    parent_chromosomes = temp_chromosome;
  }
  file.close();
  file_step.close();
  std::cout<<"\npower final"<<child_chromosomes[0]->power<<std::endl;
  new_Throughput = child_chromosomes[0]->Throughput;
  new_power = child_chromosomes[0]->power;
  return child_chromosomes[0]->gene;

}

Chromosome::Chromosome(unsigned size){
  this->size = size;
  gene = (double*)malloc(sizeof(double)*size);
  if(gene == NULL)
  {
    std::cout<<"Couldn't allocate memory for new chromozome"<<std::endl;
    exit(0);
  }
};

unsigned Chromosome::generate_random_values(unsigned min, unsigned max) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(min, max); // define the range
  return (unsigned)(distr(gen));
}


double Chromosome::power_calculation(){
  power = 0;
  for(int j=0;j<size;j++)
    power += pow(gene[j],2);
  return power;
}

struct Node* Binary_tree::create_node(double value,int constraint,unsigned idx) {
  struct Node* node;
  node = (struct Node*)malloc(sizeof(struct Node));
  node->value = value;
  node->idx = idx;
  node->constraint = constraint;
  node->right = NULL;
  node->left = NULL;
  return node;
}

void Binary_tree::insert(struct Node* root,double value,int constraint,unsigned idx){
  if(root == NULL)
    root = create_node(value,constraint,idx);
  else{
    if(value >= root->value ){
      if(root->left == NULL)
        root->left = create_node(value,constraint,idx);
      else
        insert(root->left,value,constraint,idx);
    }
    else
      if(root->right == NULL)
         root->right = create_node(value,constraint,idx);
      else
        insert(root->right,value,constraint,idx);
  }
}



void Binary_tree::create_tree(unsigned int num_nodes, class Chromosome **chromosome_tree) {
  counter = 0;
  this->num_nodes = num_nodes;
  num_nodes_with_cns_one = chromosome_tree[0]->constraint;
  root = create_node(chromosome_tree[0]->power_time,chromosome_tree[0]->constraint,0);
  this->num_nodes = num_nodes;
  for(unsigned i = 1;i<num_nodes;i++) {
    insert(root, chromosome_tree[i]->power_time, chromosome_tree[i]->constraint,
           i);
    num_nodes_with_cns_one += chromosome_tree[i]->constraint;
  }
std::cout<<"\nnumber of codes: "<<num_nodes<< "\t"<<num_nodes_with_cns_one<<std::endl;
}

void Binary_tree::create_arrays(struct Node* node,std::vector<struct array_data>&Array_con_one,std::vector<struct array_data>&Array_con_zero){
  if(node == NULL)
    return ;
//  create_arrays(node->left);
//    if(node->constraint == 1)
//  std::cout<<"print tree\t"<<node->value<<"\t"<<node->idx<<"\t"<<node->constraint<<std::endl;
//  create_arrays(node->right);

  create_arrays(node->left,Array_con_one,Array_con_zero);

  struct array_data temp;
  temp.constraint = node->constraint;
  temp.value = node->value;
  temp.idx = node->idx;

  if(node->constraint == 1) {
    std::cout << "print tree\t" << node->value << "\t" << node->idx << "\t"
              << node->constraint << std::endl;
    Array_con_one.push_back(temp);
  }
  else
    Array_con_zero.push_back(temp);
  create_arrays(node->right,Array_con_one,Array_con_zero);
}

void Binary_tree::print_(std::vector<struct array_data>Array_con_one,std::vector<struct array_data>Array_con_zero) {
  std::ofstream Arrays;
  Arrays.open("/home/pouria/Desktop/G_GPU/DATA/arrays.txt",std::ios::app);
  Arrays<<"\n*****************\n"<<"sizeof vectors: "<<Array_con_one.size()<<"\t"<<Array_con_zero.size()<<std::endl;
  for(struct array_data array: Array_con_one)
    Arrays<<"Ones idx: "<<array.idx<<"\tconstraint: "<<array.constraint<<"\tvalue: "<<array.value<<std::endl;
  for(struct array_data array: Array_con_zero)
    Arrays<<"Zeros idx: "<<array.idx<<"\tconstraint: "<<array.constraint<<"\tvalue: "<<array.value<<std::endl;
  Arrays.close();
}

struct array_data Binary_tree::find_min(std::vector<struct array_data>&Array_con_one,
                                        std::vector<struct array_data>&Array_con_zero){
  struct array_data data;
  if(Array_con_one.size() == 0) {
    data = Array_con_zero.back();
    Array_con_zero.pop_back();
  }
  else{
    data = Array_con_one.back();
    Array_con_one.pop_back();
  }
  return data;
}

void Binary_tree::delete_min_node(){
  struct Node* parent_node = root;
  if(root->left == NULL){
    root = root->right;
    return;
  }
  struct Node* left_node = root->left;
  while(left_node->left != NULL){
    parent_node = left_node;
    left_node  = left_node->left;
  }
  parent_node->left = left_node->right;
};