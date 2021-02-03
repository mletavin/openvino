/*
// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "program_impl.h"
#include "program_helpers.h"
#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>
#include <algorithm>

using namespace cldnn;

namespace {

class bits_64 {
public:
    explicit bits_64(size_t size, bool set = false) : storage((size / 64) + 1, (set ? ~0ULL : 0ULL)) {}
    bool is_set(size_t idx) const {
        size_t storage_idx = idx >> 6;
        uint64_t mask = 1ULL << (idx & 0x3F);
        return storage[storage_idx] & mask;
    }
    void set(size_t idx) {
        size_t storage_idx = idx >> 6;
        uint64_t mask = 1ULL << (idx & 0x3F);
        storage[storage_idx] |= mask;
    }
    bool _or(const bits_64& that) {
        bool changed = false;
        size_t sz = std::min(storage.size(), that.storage.size());
        for (size_t i = 0; i < sz; i++) {
            uint64_t myval = storage[i];
            uint64_t thatval = myval | that.storage[i];
            bool local_change = myval != thatval;
            changed |= local_change;
            if (local_change)
                storage[i] = thatval;
        }
        return changed;
    }
#if 0
    void dump(std::ostream& s, size_t cols) {
        size_t idx = 0;
        size_t rows = (storage.size() * 64) / cols;

        s << storage.size() << " items" << std::endl;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                s << is_set(idx);
                idx++;
            }
            s << std::endl;
        }
    }
#endif

protected:
    std::vector<uint64_t> storage;
};

}  // namespace

struct oooq_node_info {
    int order_id;
    int strand_id;
};

struct oooq_link {
    program_node* node;
    int order_id;
    int strand_id;
    size_t sz;
    program_node* parent;
};

struct oooq_strand {
    std::list<program_node*> node_steps;
    uint64_t value;
};

using oooq_list = std::list<oooq_link>;

void oooq_memory_dependencies::run(program_impl& p) {
    // For oooq memory dependencies nodes A and B can't share memory if
    // processing_num(A) < processing_num(B) and there is no path from A to B.
    // Assuming precalculation of reachability this function has complexity O(N^2 log N).

    // First create transitive closure of the graph,
    // giving us mapping of node to set of all users that can be reached from this node.
    std::vector<oooq_strand> strands;
    oooq_list input_strands;
    auto oooq_map = std::map<program_node*, oooq_node_info>();
    auto& processing_order = p.get_processing_order();

    int ni = 0;
    int global_strand_id = 0;
    for (auto& node : processing_order) {
        oooq_node_info node_info = { ni++, -1 };

        if (node->is_type<input_layout>() &&
            node->get_output_layout().format < format::oiyx) {

            //add new starting strand
            node_info.strand_id = global_strand_id++;

            oooq_link new_strand = { node, node_info.order_id, node_info.strand_id,
                                    node->get_output_layout().bytes_count(), nullptr };
            input_strands.push_back(new_strand);

            std::list<program_node*> steps = { node };
            oooq_strand strand = { steps, node->get_output_layout().bytes_count() };
            strands.push_back(strand);
        }
        oooq_map[node] = node_info;
    }

    unsigned int num_nodes = static_cast<unsigned int>(oooq_map.size());

    // full cross ref [node<->node] bitmap.
    // every node has a bit array assigned to it
    // users or the node are marked with 1 bit in this array
    std::vector<bits_64> user_bitmap(num_nodes, bits_64(num_nodes));

    // init bitmaps from direct node users
    for (const auto& node : oooq_map) {
        for (const auto& user : node.first->get_users()) {
            user_bitmap[node.second.order_id].set(oooq_map.at(user).order_id);
        }
    }

    // Iteratively extend the users set by adding closure over existing users untill no change occurs.
    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned int n = 0; n < num_nodes; n++) {
            auto& users = user_bitmap[n];

            // iterate over all users
            for (unsigned int user_id = 0; user_id < num_nodes; user_id++) {
                // if we have this user set, then add its sub-users to the map
                if (users.is_set(user_id)) {
                    changed |= users._or(user_bitmap[user_id]);
                }
            }
        }
    }

    // Connection query:
    auto are_connected = [&](unsigned int A, unsigned int B) {
        return user_bitmap[A].is_set(B);
    };

    unsigned int A = 0;
    auto itr_A = processing_order.begin();

    while (itr_A != processing_order.end()) {
        unsigned int B = ++A;
        auto itr_B = ++itr_A;
        while (itr_B != processing_order.end()) {
            if (!are_connected(A, B)) {
                add_memory_dependency(*itr_A, *itr_B);
                add_memory_dependency(*itr_B, *itr_A);
            }
            itr_B++;
            B++;
        }
    }

    //process graph step by step
    if (input_strands.empty())
        return;

    oooq_list step_list;
    int stid = 0;

    do {
        //std::cout << "\nStep " << stid++ << ", " << input_strands.size() << " inputs" << std::endl;
        for (auto& input : input_strands) {
            auto& users = input.node->get_users();

            for (auto& user : users) {
                const auto& ui = oooq_map[user];

                if (ui.strand_id != -1)      //already taken by previous step
                    continue;

                oooq_link user_info = { user, ui.order_id, ui.strand_id,
                                       user->get_output_layout().bytes_count(), input.node };
                step_list.push_back(user_info);
            }
        }

        /*std::cout << "Initial:\n";
        for (auto& step_node : step_list) {
            std::cout << "[" << step_node.order_id << "], " << step_node.node->id() << std::endl;
        };*/

        // remove duplicating nodes from future candidates
        step_list.sort([](const oooq_link& first, const oooq_link& last) -> bool {
            return first.order_id < last.order_id;
            });
        step_list.unique([](const oooq_link& first, const oooq_link& last) -> bool {
            return first.order_id == last.order_id;
            });

        /*std::cout << "after sort:\n";
        for (auto& step_node : step_list) {
            std::cout << "[" << step_node.order_id << "], " << step_node.node->id() << std::endl;
        };*/
        // remove nodes that depend on other nodes from this step -
        // they should be considered on later steps
        auto itrA = step_list.begin();
        while (itrA != step_list.end()) {
        next_candidate:
            auto itrB = itrA;
            itrB++;
            while (itrB != step_list.end()) {
                if (user_bitmap[itrB->order_id].is_set(itrA->order_id)) {
                    itrA = step_list.erase(itrA);
                    goto next_candidate;
                } else if (user_bitmap[itrA->order_id].is_set(itrB->order_id))
                    itrB = step_list.erase(itrB);
                else
                    itrB++;
            }
            itrA++;
        }
        /*std::cout << "after dep removal:\n";
        for (auto& step_node : step_list) {
            std::cout << "[" << step_node.order_id << "], " << step_node.node->id() << std::endl;
        };*/

        // find best candidate for every input to continue its strand
        int max_order_id = INT_MAX;
        for (auto& input : input_strands) {
            int min_order = max_order_id;
            auto best_output = step_list.end();
            auto itr = step_list.begin();
            while (itr != step_list.end()) {
                if (itr->parent == input.node && itr->order_id < min_order) {
                    best_output = itr;
                    min_order = itr->order_id;
                }
                itr++;
            }
            if (best_output != step_list.end()) {
                int id = input.strand_id;
                best_output->strand_id = id;
                oooq_map[best_output->node].strand_id = id;
                strands[id].node_steps.emplace_back(best_output->node);
            }
        }
        // start new strands from remaining candidates
        // and update global map
        for (auto& cand : step_list) {
            if (cand.strand_id != -1)
                continue;
            cand.strand_id = global_strand_id++;
            oooq_map[cand.node].strand_id = cand.strand_id;
            std::list<program_node*> steps = { cand.node };
            oooq_strand strand = { steps, cand.node->get_output_layout().bytes_count() };
            strands.push_back(strand);
        }

        // continue with new candidates
        input_strands.swap(step_list);
        step_list.clear();
    } while (!input_strands.empty());

    // 1) find overlapping strands,
    // try to merge their overlapping parts,
    // moving hanging heads and tails into new strands
    size_t total_merged = 0;
    //do {
        total_merged = 0;
        // sort strands by value
        std::sort(strands.begin(), strands.end(), [&](const oooq_strand& first, const oooq_strand& last) -> bool {
            return oooq_map[first.node_steps.front()].order_id < oooq_map[last.node_steps.front()].order_id; });
        stid = 0;
        //std::cout << "--------------------------------\nNext overlap iteration:" << std::endl;
        auto sA_itr = strands.begin();
        while (sA_itr != strands.end()) {
            if (sA_itr->node_steps.size()<2) {
                sA_itr++;
                //std::cout << "A is 1\n";
                continue;
            }
            auto sB_itr = sA_itr;
            sB_itr++;

            while (sB_itr != strands.end()) {
                // perform quick checks first
                if (sA_itr == sB_itr) {
                    sB_itr++;
                    //std::cout << "A==B\n";
                    continue;
                }
                if (sB_itr->node_steps.size() < 2) {
                    sB_itr++;
                    //std::cout << "B is 1\n";
                    continue;
                }

                /*std::cout << "A:\n";
                for (auto& node : sA_itr->node_steps) {
                    std::cout << "[" << oooq_map[node].order_id << "] " << node->id() << std::endl;
                }
                std::cout << "B:\n";
                for (auto& node : sB_itr->node_steps) {
                    std::cout << "[" << oooq_map[node].order_id << "] " << node->id() << std::endl;
                }*/
                std::vector<oooq_strand>::iterator first, second;
                bool swap = false;
                bool wrap = false;
                bool merged = false;

                int order_A1 = oooq_map[sA_itr->node_steps.front()].order_id;
                int order_A2 = oooq_map[sA_itr->node_steps.back()].order_id;

                int order_B1 = oooq_map[sB_itr->node_steps.front()].order_id;
                int order_B2 = oooq_map[sB_itr->node_steps.back()].order_id;

                if (user_bitmap[order_A1].is_set(order_B1)) {
                    if (user_bitmap[order_A2].is_set(order_B1)) {
                        // no overlap is possible, pick up next
                        sB_itr++;
                        //std::cout << "B head depends on A tail, continue\n";
                        continue;
                    } else {
                        //std::cout << "B head overlaps A tail\n";
                        first = sA_itr;
                        second = sB_itr;
                        if (user_bitmap[order_B2].is_set(order_A2)) {
                            //std::cout << "A wraps B\n";
                            wrap = true;
                        }
                    }
                } else if (user_bitmap[order_B1].is_set(order_A1)) {
                    if (user_bitmap[order_B2].is_set(order_A1)) {
                        // no overlap is possible, pick up next
                        sB_itr++;
                        //std::cout << "A head depends on B tail, continue\n";
                        continue;
                    } else {
                        //std::cout << "A head overlaps B tail\n";
                        first = sB_itr;
                        second = sA_itr;
                        swap = true;
                        if (user_bitmap[order_A2].is_set(order_B2)) {
                            //std::cout << "B wraps A\n";
                            wrap = true;
                        }
                    }
                } else {
                    // no overlap is possible, pick up next
                    sB_itr++;
                    //std::cout << "No dependency\n";
                    continue;
                }

                std::list<program_node*> steps;
                int tail_shift = 0, head_shift = 0;
                if (wrap) {
                    auto f_head = first->node_steps.begin();
                    auto f_tail = first->node_steps.rbegin();
                    int order_H = oooq_map[second->node_steps.front()].order_id;
                    int order_T = oooq_map[second->node_steps.back()].order_id;

                    do {
                        if (!user_bitmap[oooq_map[*f_head].order_id].is_set(order_H))
                            break;
                        f_head++;
                        head_shift++;
                    } while (f_head != first->node_steps.end());

                    do {
                        if (!user_bitmap[order_T].is_set(oooq_map[*f_tail].order_id))
                            break;
                        f_tail++;
                        tail_shift++;
                    } while (f_tail != first->node_steps.rend());

                    auto mid_sz = first->node_steps.size() - head_shift - tail_shift;
                    if (second->node_steps.size() > mid_sz) {
                        // second is bigger than middle part of the first, so swap them
                        // save middle nodes into separate list
                        auto hi = std::next(first->node_steps.begin(), head_shift);
                        for(size_t s = 0; s< mid_sz; s++) {
                            steps.push_back(*hi++);
                        }
                        // merge the pieces from first and second into first
                        second->node_steps.splice(second->node_steps.end(),
                            first->node_steps,
                            std::prev(first->node_steps.end(), tail_shift),
                            first->node_steps.end());
                        first->node_steps.resize(head_shift);
                        first->node_steps.splice(first->node_steps.end(), second->node_steps);
                        merged = true;
                    }
                } else {
                    auto f_tail = first->node_steps.rbegin();
                    auto s_head = second->node_steps.begin();
                    int order_F = oooq_map[*f_tail].order_id;
                    int order_S = oooq_map[*s_head].order_id;

                    //find head and tail shifts, see what's bigger
                    // and if it's worth swapping at all
                    do {
                        if (user_bitmap[oooq_map[*f_tail].order_id].is_set(order_S))
                            break;
                        f_tail++;
                        tail_shift++;
                    } while (f_tail != first->node_steps.rend());

                    do {
                        if (user_bitmap[order_F].is_set(oooq_map[*s_head].order_id))
                            break;
                        s_head++;
                        head_shift++;
                    } while (s_head != second->node_steps.end());

                    auto l1 = (first->node_steps.size() - tail_shift) + second->node_steps.size();
                    auto l2 = first->node_steps.size() + (second->node_steps.size() - head_shift);

                    if (second->node_steps.size() > tail_shift || first->node_steps.size() > head_shift) {
                        if (l1 > l2) {
                            // move remaining tail nodes into new strand
                            auto si = first->node_steps.rbegin();
                            do {
                                steps.push_front(*si++);
                            } while (si != f_tail);
                            //merge remainder with second
                            first->node_steps.resize(first->node_steps.size() - tail_shift);
                            first->node_steps.splice(first->node_steps.end(), second->node_steps);
                        } else {
                            // move remaining head nodes into new strand
                            auto si = second->node_steps.begin();
                            do {
                                steps.push_back(*si++);
                            } while (si != s_head);
                            //merge remainder into first
                            first->node_steps.splice(first->node_steps.end(),
                                second->node_steps,
                                std::next(second->node_steps.begin(), head_shift),
                                second->node_steps.end());
                        }
                        merged = true;
                    }
                }
                //std::cout << "head_shift " << head_shift << ", tail_shift " << tail_shift << std::endl;
                if (merged) {
                    //std::cout << "Merge\n";
                    oooq_strand strand = { steps, 0 };

                    // A will contain bigger merged strand, new smaller strand will go into B
                    if (swap)
                        *sA_itr = *sB_itr;
                    *sB_itr = strand;
                    total_merged++;
                }
                sB_itr++;
            }
            sA_itr++;
        }
        //std::cout << "Total_merged " << total_merged << std::endl;
    //} while (total_merged > 0);
    
    // 2) find and merge remaining non-overlapping, but dependent strands
    //do {
        total_merged = 0;
        // sort strands by value
        std::sort(strands.begin(), strands.end(), [&](const oooq_strand& first, const oooq_strand& last) -> bool {
            return oooq_map[first.node_steps.front()].order_id < oooq_map[last.node_steps.front()].order_id; });
        
        sA_itr = strands.begin();
        while (sA_itr != strands.end()) {
            auto sB_itr = sA_itr;
            sB_itr++;

            while (sB_itr != strands.end()) {
                if (sA_itr == sB_itr) {
                    sB_itr++;
                    //std::cout << "A==B\n";
                    continue;
                }

                int order_A1 = oooq_map[sA_itr->node_steps.front()].order_id;
                int order_A2 = oooq_map[sA_itr->node_steps.back()].order_id;
                int order_B1 = oooq_map[sB_itr->node_steps.front()].order_id;
                int order_B2 = oooq_map[sB_itr->node_steps.back()].order_id;

                /*std::cout << "A:\n";
                for (auto& node : sA_itr->node_steps) {
                    std::cout << "[" << oooq_map[node].order_id << "] " << node->id() << std::endl;
                }
                std::cout << "B:\n";
                for (auto& node : sB_itr->node_steps) {
                    std::cout << "[" << oooq_map[node].order_id << "] " << node->id() << std::endl;
                }*/
                if (user_bitmap[order_A2].is_set(order_B1)) {
                    // B head depends on A tail
                    //std::cout << "Remain merge, erase B" << std::endl;
                    sA_itr->node_steps.splice(sA_itr->node_steps.end(), sB_itr->node_steps);
                    sB_itr = strands.erase(sB_itr);
                    total_merged++;
                } else if (user_bitmap[order_B2].is_set(order_A1)) {
                    // A head depends on B tail
                    //std::cout << "Remain merge, erase A" << std::endl;
                    sB_itr->node_steps.splice(sB_itr->node_steps.end(), sA_itr->node_steps);
                    *sA_itr = *sB_itr;
                    sB_itr = strands.erase(sB_itr);
                    total_merged++;
                } else {
                    // no dependency, pick up next
                    sB_itr++;
                }
            }
            sA_itr++;
        }
        //std::cout << "Total_merged " << total_merged << std::endl;
    //} while (total_merged > 0);

    // calculate strand statistics and pre-allocate memory
    memory_strands pool_strands(strands.size());
    bool can_use_device_mem = p.get_engine().supports_allocation(allocation_type::usm_device);
    std::cout << "\nFinal:" << std::endl;
    uint64_t total_sz = 0;
    for (int i = 0; i < strands.size(); i++) {
        std::cout << "=============================================================================" << std::endl;
        std::cout << "Strand " << stid++ << ", nodes " << strands[i].node_steps.size() << std::endl;
        auto& si = pool_strands[i];
        size_t sz_padded=0;
        unsigned int turn = 0, lockable_turn = 0, total_padded = 0;
        for (auto& node : strands[i].node_steps) {
            auto layout = node->get_output_layout();
            auto sz = layout.bytes_count();
            std::cout << std::endl << sz << " : [" << oooq_map[node].order_id << "] " << node->id();
            if (node->is_padded()) {
                std::cout << ", padding {";
                std::cout << layout.data_padding.lower_size();
                std::cout << layout.data_padding.upper_size();
                std::cout << "}";
                sz_padded += sz;
                node->set_strand_info(i, total_padded);
                total_padded++;
            } else if (!can_use_device_mem || node->need_lockable_memory()) {
                if (lockable_turn & 0x01) {
                    if (sz > si.sz_odd_lock)
                        si.sz_odd_lock = sz;
                } else {
                    if (sz > si.sz_even_lock)
                        si.sz_even_lock = sz;
                }
                node->set_strand_info(i, lockable_turn);
                lockable_turn++;
            } else {
                if (turn & 0x01) {
                    if (sz > si.sz_odd)
                        si.sz_odd = sz;
                } else {
                    if (sz > si.sz_even)
                        si.sz_even = sz;
                }
                node->set_strand_info(i, turn);
                turn++;
            }
        }
        std::cout << std::endl << total_padded << " padded nodes, " << sz_padded << ", bytes total\n";
        std::cout << "Non-padded alloc:\n";
        std::cout << turn << " device nodes, " << si.sz_even << " + " << si.sz_odd << " bytes\n";
        std::cout << lockable_turn << " lockable nodes, " << si.sz_even_lock << " + " << si.sz_odd_lock << " bytes\n";
        std::cout << "=============================================================================" << std::endl;
        total_sz += si.sz_even + si.sz_odd + si.sz_even_lock + si.sz_odd_lock;
    }
    std::cout << "\nTotal non-padded: " << total_sz << std::endl;
    p.get_engine().get_memory_pool().prepare_strands(p.get_id(), pool_strands);
}
