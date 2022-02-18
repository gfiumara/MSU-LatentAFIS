/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   include.h
 * Author: cori
 *
 * Created on November 4, 2018, 4:37 PM
 */

#ifndef INCLUDE_H
#define INCLUDE_H

#include <cmath>
#include <numeric>
#include <vector>

#define PI 3.1415926

class MinuPoint
{
    public:
        int x;
        int y;
        float ori;
        float reliability;
};

enum TemplateType {Texture,Minutiae};
enum FPType {Latent,Rolled};

//template base class
//used for latent and rolled minu templates in the original matcher
class SingleTemplate
{
    public:
        int m_nrof_feature{0};
        int m_des_length{0};
        int m_block_size{16};
        int m_blkH{0};
        int m_blkW{0};
        std::vector<float> m_des{};
        std::vector<MinuPoint> m_minutiae{};
        TemplateType m_template_type{TemplateType::Minutiae};
        SingleTemplate() = default;

	SingleTemplate(const TemplateType type) : m_template_type{type}{};

        SingleTemplate(const int nrof_minutiae, const int des_length):
            m_des_length(des_length),
            m_des(nrof_minutiae*m_des_length),
            m_minutiae(nrof_minutiae)
        {
        };

        SingleTemplate(const int nrof_minutiae, const int des_length, const std::vector<float> des, const int blkH, const int blkW):
        m_des_length{des_length},
        m_blkH{blkH},
        m_blkW{blkW},
        m_des{des},
        m_minutiae(nrof_minutiae)
        {
        }

        SingleTemplate(const int nrof_minutiae, const int des_length, const std::vector<float> des):
        m_des_length{des_length},
        m_des{des},
        m_minutiae(nrof_minutiae)
        {
        }

        void set_x(const std::vector<short> &x)
        {
            for(int i=0;i<this->m_minutiae.size(); ++i)
            {
                m_minutiae[i].x = x[i];
            }
        }
        void set_y(const std::vector<short> &y)
        {
            for(int i=0;i<this->m_minutiae.size(); ++i)
            {
                m_minutiae[i].y = y[i];
            }
        };
        void set_ori(const std::vector<float> &ori)
        {
            for(int i=0;i<this->m_minutiae.size(); ++i)
            {
                m_minutiae[i].ori = ori[i];
            }
        };
        void set_type(const TemplateType template_type){
            m_template_type = template_type;
        };


};

class MinutiaeTemplate:public SingleTemplate{
    public:
        MinutiaeTemplate() : SingleTemplate()
        {
        };

        MinutiaeTemplate(const int nrof_minutiae, const int des_length):SingleTemplate(nrof_minutiae, des_length)
        {
        };

        // minutiae, minutiae descriptor and ridge flow are included in minutiae template
        MinutiaeTemplate(const int nrof_minutiae, const std::vector<short> &x,const std::vector<short> &y,const std::vector<float> &ori,const int des_length, const std::vector<float> &des, const int blkH, const int blkW):
        SingleTemplate(nrof_minutiae, des_length, des, blkH, blkW)
        {
            // minutiae
            set_x(x);
            set_y(y);
            set_ori(ori);
        };

};

class TextureTemplate: public SingleTemplate
{
    public:
        TextureTemplate() : SingleTemplate(TemplateType::Texture)
        {
        };
        TextureTemplate(const int nrof_minutiae, const int des_length):SingleTemplate(nrof_minutiae, des_length)
        {
        	this->set_type(TemplateType::Texture);
        };

        // Only minutiae and minutiae descriptors are included in texture template
        TextureTemplate(const int nrof_minutiae, const std::vector<short> &x,const std::vector<short> &y,const std::vector<float> &ori, const int des_length, const std::vector<float> &des):
        SingleTemplate(nrof_minutiae, des_length, des)
        {
            this->set_type(TemplateType::Texture);

            set_x(x);
            set_y(y);
            set_ori(ori);
        };
};

class LatentTextureTemplate: public TextureTemplate
{
    public:
        std::vector<float> m_dist_codewords{};
        LatentTextureTemplate() : TextureTemplate()
        {
        };

        LatentTextureTemplate(const int nrof_minutiae, const int des_length):TextureTemplate(nrof_minutiae,des_length)
        {
        };
        LatentTextureTemplate(const int nrof_minutiae, const std::vector<short> &x,const std::vector<short> &y,const std::vector<float> &ori, const int des_length, const std::vector<float> &des):
        TextureTemplate(nrof_minutiae, x, y, ori, des_length, des)
        {}

        void compute_dist_to_codewords(const std::vector<float> &codewords, const int nrof_subs, const int sub_dim,  const int nrof_clusters)
        {
            m_dist_codewords.resize(this->m_minutiae.size()*nrof_subs*nrof_clusters);

            float *pword2;
            for(int i=0; i<this->m_minutiae.size() ; ++i)
            {
                float *pdes0 = m_des.data() +  i*m_des_length;

                for(int j=0;j<nrof_subs ; ++j)
                {
                    const float *pdes1 = pdes0 + j*sub_dim;
                    const float *pword0 = codewords.data() + j*nrof_clusters*sub_dim;
                    const float min_v = 1000;
                    for(int q=0; q<nrof_clusters; ++q)
                    {
                        const float *pword1 = pword0 + q*sub_dim;
                        float dist = 0.0;
                        for(int k=0; k<sub_dim; ++k)
                        {
                            const float *pword2 = pword1 + k;
                            const float *pdes2 = pdes1 + k;
                            dist += (*pdes2-*pword2)* (*pdes2-*pword2);
                        }
                        m_dist_codewords[i*nrof_subs*nrof_clusters+j*nrof_clusters+q] = (dist);
                    }

                }
            }


        };
};

class RolledTextureTemplatePQ:public TextureTemplate
{
    public:
        std::vector<unsigned char> m_desPQ{};
        RolledTextureTemplatePQ() : TextureTemplate() {}

        RolledTextureTemplatePQ(const int nrof_minutiae, const int des_length):TextureTemplate(nrof_minutiae, des_length),
            m_desPQ(nrof_minutiae*m_des_length)
        {
        }

        RolledTextureTemplatePQ(const int nrof_minutiae, const std::vector<short> &x,const std::vector<short> &y,const std::vector<float> &ori, const int des_length, const std::vector<float> &des):
        TextureTemplate(nrof_minutiae, x, y, ori, des_length, {})
        {
        	/* FIXME: m_desPQ is unsigned char, but storing floats? Bug? */
        	for (const float &f : des)
        		m_desPQ.emplace_back(static_cast<unsigned char>(f));
        }
};

class FPTemplate
{
    public:
        std::vector<MinutiaeTemplate> m_minu_templates{};
        FPType m_FP_type;

        FPTemplate(const FPType FP_type):m_FP_type(FP_type)
        {
        };
        void add_template(const MinutiaeTemplate & minutiae_template)
        {
            m_minu_templates.push_back(minutiae_template);
        };
};

class LatentFPTemplate:public FPTemplate{
public:
    std::vector<LatentTextureTemplate> m_texture_templates{};
    LatentFPTemplate():FPTemplate(Latent){};

    void add_texture_template(const LatentTextureTemplate &texture_template)
    {
        m_texture_templates.push_back(texture_template);
    };

};

class RolledFPTemplate:public FPTemplate{
public:
    std::vector<RolledTextureTemplatePQ> m_texture_templates{};
    RolledFPTemplate():FPTemplate(Rolled){};

    void add_texture_template(const RolledTextureTemplatePQ & texture_template)
    {
        RolledTextureTemplatePQ texture_template_new(texture_template);
        m_texture_templates.push_back(texture_template_new);
    };
};


#endif /* INCLUDE_H */
