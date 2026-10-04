#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
namespace rp {
struct Tokenizer{int vocab=260; int bos=256,eos=257,user=258,assistant=259;
 std::vector<int> encode(const std::string&s)const{std::vector<int>t{bos};for(unsigned char c:s)t.push_back((int)c);t.push_back(eos);return t;}
 std::string decode(const std::vector<int>&t)const{std::string s;for(int x:t)if(x<256)s.push_back((char)x);return s;}
};
}
