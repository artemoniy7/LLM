#pragma once
#include "tensor.hpp"
#include <fstream>
#include <random>
#include <string>
#include <vector>
namespace rp {
struct Config{int vocab=260,ctx=128,d=128,hidden=384,layers=3;};
struct Block{Tensor wq,wk,wv,wo,w1,w2,b1,b2;};
struct Model{
 Config c; std::mt19937 rng; Tensor tok,pos,head; std::vector<Block>b;
 Model(Config cfg={},unsigned seed=42):c(cfg),rng(seed){init();}
 void init(){float e=1.f/std::sqrt((float)c.d);tok=param({c.vocab,c.d},rng,e);pos=param({c.ctx,c.d},rng,e);head=param({c.d,c.vocab},rng,e);for(int l=0;l<c.layers;++l){Block x; x.wq=param({c.d,c.d},rng,e);x.wk=param({c.d,c.d},rng,e);x.wv=param({c.d,c.d},rng,e);x.wo=param({c.d,c.d},rng,e);x.w1=param({c.d,c.hidden},rng,e);x.w2=param({c.hidden,c.d},rng,std::sqrt(2.f/c.hidden));x.b1=param({c.hidden},rng,0);x.b2=param({c.d},rng,0);b.push_back(x);}}
 Tensor embedding(const std::vector<int>&ids){auto o=make({(int)ids.size(),c.d},true);for(int i=0;i<(int)ids.size();++i)for(int j=0;j<c.d;++j)o->x[i*c.d+j]=tok->x[ids[i]*c.d+j]+pos->x[i*c.d+j];o->p={tok,pos};o->bw=[this,o,ids](){for(int i=0;i<(int)ids.size();++i)for(int j=0;j<c.d;++j){tok->g[ids[i]*c.d+j]+=o->g[i*c.d+j];pos->g[i*c.d+j]+=o->g[i*c.d+j];}};return o;}
 Tensor forward(const std::vector<int>&ids){Tensor x=embedding(ids);for(auto&z:b){auto n=layernorm(x);auto q=matmul(n,z.wq),k=matmul(n,z.wk),v=matmul(n,z.wv);x=add(x,matmul(attention(q,k,v),z.wo));n=layernorm(x);auto h=gelu(add_bias(matmul(n,z.w1),z.b1));x=add(x,add_bias(matmul(h,z.w2),z.b2));}return matmul(layernorm(x),head);}
 std::vector<Tensor> params(){std::vector<Tensor>p{tok,pos,head};for(auto&z:b)p.insert(p.end(),{z.wq,z.wk,z.wv,z.wo,z.w1,z.w2,z.b1,z.b2});return p;}
 void save(const std::string&f){std::ofstream o(f,std::ios::binary);o.write((char*)&c,sizeof(c));auto p=params();for(auto&t:p){size_t n=t->x.size();o.write((char*)&n,sizeof(n));o.write((char*)t->x.data(),n*sizeof(float));}}
 bool load(const std::string&f){std::ifstream i(f,std::ios::binary);if(!i)return false;Config old;i.read((char*)&old,sizeof(old));if(old.vocab!=c.vocab||old.ctx!=c.ctx||old.d!=c.d||old.hidden!=c.hidden||old.layers!=c.layers)return false;auto p=params();for(auto&t:p){size_t n;i.read((char*)&n,sizeof(n));if(n!=t->x.size())return false;i.read((char*)t->x.data(),n*sizeof(float));}return true;}
};
}
