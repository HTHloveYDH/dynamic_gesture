#ifndef HUNGARAINTRACK_H
#define HUNGARAINTRACK_H

#include "util/hungarian_optimizer.h"

/**
 * @brief Update the costs matrix of hungarian optimizer.
 *
 * @param association_mat The association matrix of tracks and objects, which
 * represents the bipartite graph to be optimized.
 * @param costs The costs matrix of hungarian optimizer.
 */
void UpdateCosts(const std::vector<std::vector<float>>& association_mat, SecureMat<float>* costs);

/**
 * @brief Print the assignments result.
 *
 * @param assignments Assignments result to be printed.
 */
void PrintAssignments(const std::vector<std::pair<size_t, size_t>>& assignments);

/**
 * @brief 
 * 
 * @param box 
 * @param afterNMSBoxVector 
 * @param func 
 * @return std::vector<std::pair<size_t, size_t>> 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
std::vector<std::pair<size_t, size_t>> hungarianTrack(std::vector<float> &box, \
                                                      std::vector<std::vector<float>> &afterNMSBoxVector, \
                                                      float (*func)(std::vector<float> box1, std::vector<float> box2));

#endif
