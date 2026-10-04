#pragma once
#include "tensor.hpp"
#include <cmath>
#include <fstream>
#include <vector>
namespace rp {struct AdamW{float lr=3e-4f,b1=.9f,b2=.999f,wd=.01f;long long step=0;std::vector<std::vector<float>>m,v;void init(const std::vector<Tensor>&p){m.clear();v.clear();for(auto&t:p){m.emplace_back(t->x.size(),0);v.emplace_back(t->x.size(),0);}}void update(const std::vector<Tensor>&p){++step;float bc1=1-std::pow(b1,(float)step),bc2=1-std::pow(b2,(float)step);for(size_t k=0;k<p.size();++k){auto&t=p[k];for(size_t i=0;i<t->x.size();++i){float g=t->g[i]+wd*t->x[i];m[k][i]=b1*m[k][i]+(1-b1)*g;v[k][i]=b2*v[k][i]+(1-b2)*g*g;t->x[i]-=lr*(m[k][i]/bc1)/(std::sqrt(v[k][i]/bc2)+1e-8f);t->g[i]=0;}}}};
inline void clip(const std::vector<Tensor>&p,float mx){double z=0;for(auto&t:p)for(float g:t->g)z+=g*g;float n=std::sqrt(z);if(n>mx){float q=mx/n;for(auto&t:p)for(float&g:t->g)g*=q;}}
}
