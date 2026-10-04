#pragma once
#include "tensor.hpp"
#include <cmath>
#include <vector>
namespace rp {
struct AdamW {
    float lr=3e-4f,b1=.9f,b2=.999f,wd=.01f;
    long long step=0;
    std::vector<std::vector<float>> m,v;
    void init(const std::vector<Tensor>&p) {
        m.clear(); v.clear();
        for (auto&t:p) {
            m.emplace_back(t->x.size(), 0.0f);
            v.emplace_back(t->x.size(), 0.0f);
        }
    }
    void update(const std::vector<Tensor>&p) {
        ++step;
        float bc1=1.0f-static_cast<float>(std::pow(b1,static_cast<float>(step)));
        float bc2=1.0f-static_cast<float>(std::pow(b2,static_cast<float>(step)));
        for(size_t k=0;k<p.size();++k){
            auto&t=p[k];
            for(size_t i=0;i<t->x.size();++i){
                float g=t->g[i]+wd*t->x[i];
                m[k][i]=b1*m[k][i]+(1.0f-b1)*g;
                v[k][i]=b2*v[k][i]+(1.0f-b2)*g*g;
                t->x[i]-=lr*(m[k][i]/bc1)/(std::sqrt(v[k][i]/bc2)+1e-8f);
                t->g[i]=0.0f;
            }
        }
    }
};
inline void clip(const std::vector<Tensor>&p,float mx){
    double z=0.0;
    for(auto&t:p)for(float g:t->g)z+=static_cast<double>(g)*g;
    float n=static_cast<float>(std::sqrt(z));
    if(n>mx){float q=mx/n;for(auto&t:p)for(float&g:t->g)g*=q;}
}
}
