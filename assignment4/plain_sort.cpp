/*
 * ----------------------------------------------------------------------------
 *  File: plain_sort.cpp
 *  Description:
 *      Sequential baseline program that reads a binary dataset of fixed‑size
 *      records and sorts them entirely in memory with a single call to
 *      std::sort. Intended as a performance reference for parallel versions.
 *
 *      Build:
 *          g++ -std=c++17 -O3 -march=native plain_sort.cpp -o plain_sort
 *
 *      Example run:
 *          ./plain_sort -s 100M -r 256
 *
 *  Author: <Giuseppe Gabriele Russo>
 *  Date:   2025-06-03
 * ----------------------------------------------------------------------------
 */
// =======================================================================
//  plain_sort.cpp  –  Sequential baseline (single std::sort)
// =======================================================================
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#ifndef RPAYLOAD
#define RPAYLOAD 64 // default fallback
#endif

/**
 * @brief Plain old data structure representing one record.
 *
 * Contains the 64‑bit key used for ordering and a fixed‑size payload
 * buffer whose length is set at compile‑time via RPAYLOAD.
 */
struct Record
{
    unsigned long key;
    char rpayload[RPAYLOAD];
};
static std::size_t PAYLOAD_SIZE = 64;

/**
 * @brief Load the entire binary dataset from disk.
 *
 * File layout:
 *   uint64_t N   – number of records
 *   uint32_t P   – payload size in bytes
 *   uint32_t pad – ignored
 *   N × (uint64_t key + P-byte payload)
 *
 * @param f  Path to the dataset file.
 * @return   Vector of Record objects loaded into memory.
 */
std::vector<Record> load_records(const std::string &f)
{
    std::ifstream in(f, std::ios::binary);
    if (!in)
        throw std::runtime_error("open file");
    uint64_t N;
    uint32_t P, pad;
    in.read((char *)&N, 8);
    in.read((char *)&P, 4);
    in.read((char *)&pad, 4);
    PAYLOAD_SIZE = P;
    std::vector<Record> v(N);
    for (uint64_t i = 0; i < N; ++i)
    {
        in.read((char *)&v[i].key, 8);
        in.read(v[i].rpayload, P);
    }
    return v;
}




void write_sorted_records(const std::string& outname, const std::vector<Record>& recs) {
    std::ofstream out(outname, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot open output file");

    uint64_t N = recs.size();
    uint32_t P = static_cast<uint32_t>(PAYLOAD_SIZE);
    uint32_t pad = 0;

    // Scrive header: [N | P | pad]
    out.write(reinterpret_cast<const char*>(&N), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(&P), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&pad), sizeof(uint32_t));

    // Scrive i record ordinati
    for (const auto& r : recs) {
        out.write(reinterpret_cast<const char*>(&r.key), 8);
        out.write(r.rpayload, P);
    }
}








/**
 * @brief Parse a human‑readable size string with optional K/M/G suffix.
 *
 * Examples: "100M" → 100 000 000, "64" → 64.
 *
 * @param s  String to parse.
 * @return   Numeric value expressed in bytes.
 */
// -----------------------------------------------------------------------
// Parse size suffixes K/M/G -----------------------------------------------------
std::size_t parse_size(std::string s){
    std::size_t mult=1; char c=s.back();
    if (c=='K'||c=='k') mult=1'000ULL;
    else if (c=='M'||c=='m') mult=1'000'000ULL;
    else if (c=='G'||c=='g') mult=1'000'000'000ULL;
    if (!std::isdigit(c)) s.pop_back();
    return std::stoull(s)*mult;
}

/**
 * @brief Program entry point: parses CLI options, loads the dataset,
 *        performs sequential std::sort, prints timings, and optionally
 *        verifies the sorted output.
 */
// =======================================================================
int main(int argc,char* argv[]){
    std::size_t N=0;
    PAYLOAD_SIZE=8;

    std::string s_arg, r_arg;
    for(int i=1;i<argc;++i){
        if (!strcmp(argv[i],"-s") && i+1<argc) {
            s_arg = argv[++i];                 
            N = parse_size(s_arg);
        }
        else if (!strcmp(argv[i],"-r") && i+1<argc) {
            r_arg = argv[++i];                 
            PAYLOAD_SIZE = parse_size(r_arg);
        }
    }
    if(!N){ std::cerr<<"Usage: "<<argv[0]<<" -s <size> [-r payload]\n"; return 1;}

    std::string fname = "./Dataset/data_" + s_arg + "_" + r_arg + ".bin";

    auto t_start = std::chrono::high_resolution_clock::now();
    auto localRecs = load_records(fname.c_str());

    //std::cout<<"Sequential std::sort on "<<N<<" records (payload "<<PAYLOAD_SIZE<<" B)…\n";
    auto t_mid = std::chrono::high_resolution_clock::now();
    std::sort(localRecs.begin(), localRecs.end(),
              [](const Record& a,const Record& b){ return a.key<b.key; });
    auto t_mid2 = std::chrono::high_resolution_clock::now();



    // ==================== [TEST CODE BEGIN] ====================
    for(std::size_t i=1;i<N;++i) 
        if(localRecs[i-1].key>localRecs[i].key){ std::cerr<<"Not sorted!\n"; break;}
    // ==================== [TEST CODE END] ======================


    std::string outname = "./Dataset/sorted_plain_" + s_arg + "_" + r_arg + ".bin";
    write_sorted_records(outname, localRecs);
    auto t_end = std::chrono::high_resolution_clock::now();


    std::cout<< "Total (get data) time: "
             <<std::chrono::duration<double>(t_mid-t_start).count()<<" s\n";
    std::cout<< "Total (sort) time: "
             <<std::chrono::duration<double>(t_mid2-t_mid).count()<<" s\n";
    std::cout<< "Total (write) time: "
             <<std::chrono::duration<double>(t_end-t_mid2).count()<<" s\n";
    std::cout<< "TOTAL TIME: "
             <<std::chrono::duration<double>((t_mid-t_start)+
                                              (t_mid2-t_mid)+
                                              (t_end-t_mid2)).count()<<" s\n";


    return 0;
}