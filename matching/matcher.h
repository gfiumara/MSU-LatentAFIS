/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   matcher_int.h
 * Author: cori
 *
 * Created on November 4, 2018, 4:49 PM
 */

#ifndef MATCHER_INT_H
#define MATCHER_INT_H
#include <cstdlib>
#include <type_traits>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include"include.h"

using namespace std;
using namespace std::chrono;
#define PI 3.1415926
#define EIGEN_DONT_PARALLELIZE

namespace PQ
{
    const unsigned long MaxNRolledMinu = 1000;
    const unsigned long MaxNLatentMinu = 1000;

class Matcher final {


public:
    Matcher(string code_file);

    int List2List_matching(string latent_list_file, string rolled_list_file, string output_file);
    int One2List_matching(string latent_template_file, string rolled_list_file, string score_file);
    int One2One_matching(string latent_file, string rolled_file);

    float One2One_minutiae_matching(const MinutiaeTemplate  &latent_minu_template, const MinutiaeTemplate  &rolled_minu_template, bool save_corr = false, string corr_file = "") const;
    float One2One_texture_matching(const LatentTextureTemplate &latent_texture_template, const RolledTextureTemplatePQ &rolled_minu_template) const;
    int One2One_matching_selected_templates(const LatentFPTemplate &latent_template, const RolledFPTemplate &rolled_template, vector<float> & score, bool save_corr = false, string corr_file = "") const;
    int One2One_matching_all_templates(const LatentFPTemplate &latent_template, const RolledFPTemplate &rolled_template, vector<float> & score);

    LatentFPTemplate load_latent_template(const string &tname) const;
    RolledFPTemplate load_rolled_template(const string &tname) const;
    LatentFPTemplate load_latent_template(const std::vector<uint8_t> &buf) const;
    RolledFPTemplate load_rolled_template(const std::vector<uint8_t> &buf) const;

private:
    vector<tuple<float, int, int>>  LSS_R_Fast2_Dist(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template,const SingleTemplate & rolled_template, float d_thr=30.0) const;
    vector<tuple<float, int, int>>  LSS_R_Fast2_Dist_eigen(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, float d_thr = 30.0) const;
    vector<tuple<float, int, int>>  LSS_R_Fast2_Dist_lookup(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, float d_thr = 30.0) const;
    vector<tuple<float, int, int>>  LSS_R_Fast2(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, int d_thr=3) const;
    float adjust_angle(float angle) const;

private:
    int N; // top N minutiae correspondences for matching

    vector<string> description;
    vector<float> table_dist;
    int dist_N;
    int max_nrof_templates;
    int nrof_subs;
    int nrof_clusters;
    int sub_dim;
    std::vector<float> codewords{};
};
}
#endif /* MATCHER_H */
