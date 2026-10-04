#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace rp {
struct Node{std::vector<float>x,g;std::vector<int>s;bool req;std::vector<std::shared_ptr<Node>> p;std::function<void()> bw;Node(std::vector<int>sh,bool r):s(std::move(sh)),req(r){size_t n=1;for(int v:s)n*=v;x.resize(n);if(req)g.assign(n,0);}};
using Tensor=std::shared_ptr<Node>;
inline Tensor make(std::vector<int>s,bool r=false){return std::make_shared<Node>(std::move(s),r);} inline Tensor param(std::vector<int>s,std::mt19937&rng,float sc){auto t=make(s,true);std::normal_distribution<float>d(0,sc);for(auto&v:t->x)v=d(rng);return t;} inline void zero_grad(Tensor&t){if(t->req)std::fill(t->g.begin(),t->g.end(),0);}
inline Tensor add(Tensor a,Tensor b){if(a->x.size()!=b->x.size())throw std::runtime_error("add shape");auto o=make(a->s,a->req||b->req);for(size_t i=0;i<o->x.size();++i)o->x[i]=a->x[i]+b->x[i];if(o->req){o->p={a,b};o->bw=[a,b,o](){for(size_t i=0;i<o->x.size();++i){if(a->req)a->g[i]+=o->g[i];if(b->req)b->g[i]+=o->g[i];}};}return o;}
inline Tensor add_bias(Tensor a,Tensor b){int T=a->s[0],D=a->s[1];auto o=make(a->s,a->req||b->req);for(int i=0;i<T;++i)for(int j=0;j<D;++j)o->x[i*D+j]=a->x[i*D+j]+b->x[j];if(o->req){o->p={a,b};o->bw=[a,b,o,T,D](){for(int i=0;i<T;++i)for(int j=0;j<D;++j){float z=o->g[i*D+j];if(a->req)a->g[i*D+j]+=z;if(b->req)b->g[j]+=z;}};}return o;}
inline Tensor mul_scalar(Tensor a,float c){auto o=make(a->s,a->req);for(size_t i=0;i<o->x.size();++i)o->x[i]=a->x[i]*c;if(o->req){o->p={a};o->bw=[a,o,c](){for(size_t i=0;i<o->x.size();++i)a->g[i]+=o->g[i]*c;};}return o;}
inline Tensor matmul(Tensor a,Tensor b){int m=a->s[0],k=a->s[1],n=b->s[1];auto o=make({m,n},a->req||b->req);for(int i=0;i<m;++i)for(int j=0;j<n;++j)for(int q=0;q<k;++q)o->x[i*n+j]+=a->x[i*k+q]*b->x[q*n+j];if(o->req){o->p={a,b};o->bw=[a,b,o,m,k,n](){for(int i=0;i<m;++i)for(int j=0;j<n;++j){float z=o->g[i*n+j];for(int q=0;q<k;++q){if(a->req)a->g[i*k+q]+=z*b->x[q*n+j];if(b->req)b->g[q*n+j]+=z*a->x[i*k+q];}}};}return o;}
inline Tensor gelu(Tensor a){auto o=make(a->s,a->req);const float c=.79788456f;for(size_t i=0;i<o->x.size();++i){float z=a->x[i],u=c*(z+.044715f*z*z*z);o->x[i]=.5f*z*(1+std::tanh(u));}if(o->req){o->p={a};o->bw=[a,o,c](){for(size_t i=0;i<o->x.size();++i){float z=a->x[i],u=c*(z+.044715f*z*z*z),th=std::tanh(u),du=c*(1+.134145f*z*z);a->g[i]+=o->g[i]*(.5f*(1+th)+.5f*z*(1-th*th)*du);}};}return o;}
inline Tensor layernorm(Tensor a,float eps=1e-5f){int T=a->s[0],D=a->s[1];auto o=make(a->s,a->req);for(int i=0;i<T;++i){float mu=0,v=0;for(int j=0;j<D;++j)mu+=a->x[i*D+j];mu/=D;for(int j=0;j<D;++j){float z=a->x[i*D+j]-mu;v+=z*z;}v/=D;float inv=1/std::sqrt(v+eps);for(int j=0;j<D;++j)o->x[i*D+j]=(a->x[i*D+j]-mu)*inv;}if(o->req){o->p={a};o->bw=[a,o,T,D,eps](){for(int i=0;i<T;++i){float mu=0,v=0;for(int j=0;j<D;++j)mu+=a->x[i*D+j];mu/=D;for(int j=0;j<D;++j){float z=a->x[i*D+j]-mu;v+=z*z;}v/=D;float inv=1/std::sqrt(v+eps),sg=0,szg=0;for(int j=0;j<D;++j){float z=a->x[i*D+j]-mu;sg+=o->g[i*D+j];szg+=o->g[i*D+j]*z;}for(int j=0;j<D;++j){float z=a->x[i*D+j]-mu;a->g[i*D+j]+=inv*(o->g[i*D+j]-sg/D-z*szg/(D*(v+eps)));}}};}return o;}
inline Tensor softmax(Tensor a){int T=a->s[0],N=a->s[1];auto o=make(a->s,a->req);for(int i=0;i<T;++i){float mx=-1e30f,z=0;for(int j=0;j<N;++j)mx=std::max(mx,a->x[i*N+j]);for(int j=0;j<N;++j){o->x[i*N+j]=std::exp(a->x[i*N+j]-mx);z+=o->x[i*N+j];}for(int j=0;j<N;++j)o->x[i*N+j]/=z;}if(o->req){o->p={a};o->bw=[a,o,T,N](){for(int i=0;i<T;++i){float dot=0;for(int j=0;j<N;++j)dot+=o->g[i*N+j]*o->x[i*N+j];for(int j=0;j<N;++j)a->g[i*N+j]+=o->x[i*N+j]*(o->g[i*N+j]-dot);}};}return o;}
inline Tensor transpose(Tensor a){int m=a->s[0],n=a->s[1];auto o=make({n,m},a->req);for(int i=0;i<m;++i)for(int j=0;j<n;++j)o->x[j*m+i]=a->x[i*n+j];if(o->req){o->p={a};o->bw=[a,o,m,n](){for(int i=0;i<m;++i)for(int j=0;j<n;++j)a->g[i*n+j]+=o->g[j*m+i];};}return o;}
inline Tensor attention(Tensor q,Tensor k,Tensor v){int T=q->s[0],D=q->s[1];auto scores=mul_scalar(matmul(q,transpose(k)),1.0f/std::sqrt((float)D));for(int i=0;i<T;++i)for(int j=i+1;j<T;++j)scores->x[i*T+j]=-1e9f;auto p=softmax(scores);return matmul(p,v);}
inline float loss_and_seed(Tensor logits,const std::vector<int>& y){int T=logits->s[0],V=logits->s[1];float loss=0;for(int i=0;i<T;++i){float mx=-1e30f,z=0;for(int j=0;j<V;++j)mx=std::max(mx,logits->x[i*V+j]);for(int j=0;j<V;++j)z+=std::exp(logits->x[i*V+j]-mx);loss+=-logits->x[i*V+y[i]]+mx+std::log(z);for(int j=0;j<V;++j)logits->g[i*V+j]=(std::exp(logits->x[i*V+j]-mx)/z-(j==y[i]))/T;}return loss/T;}
inline void backward(Tensor root){std::vector<Tensor> order,st{root};std::unordered_set<Node*>seen;while(!st.empty()){auto t=st.back();st.pop_back();if(!seen.insert(t.get()).second)continue;order.push_back(t);for(auto&p:t->p)st.push_back(p);}for(auto it=order.rbegin();it!=order.rend();++it)if((*it)->bw)(*it)->bw();}
}
