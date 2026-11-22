/* Fixed 245-tick A/B test of an additional decoded pre-cut picture in libaom. */
#include <aom/aom_encoder.h>
#include <aom/aomcx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int valid, const char *message) {
    if (!valid) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}
static void little(FILE *f, unsigned long long value, int length) {
    for (int i = 0; i < length; ++i) {
        require(fputc((int)(value & 255), f) != EOF, "IVF write failed");
        value >>= 8;
    }
}
int main(int argc, char **argv) {
    require(argc == 5, "ref input.yuv fresh-output.ivf Q enable-long-reference");
    const int width = 320, height = 180, count = 245;
    int qp = atoi(argv[3]), long_ref = atoi(argv[4]);
    require(qp >= 0 && qp <= 63 && (long_ref == 0 || long_ref == 1), "arguments");
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wbx");
    require(in && out, "input or exclusive output");
    aom_codec_ctx_t encoder = {0};
    aom_codec_enc_cfg_t cfg;
    aom_codec_iface_t *iface = aom_codec_av1_cx();
    require(!aom_codec_enc_config_default(iface, &cfg, AOM_USAGE_REALTIME), "config");
    cfg.g_w = width; cfg.g_h = height; cfg.g_threads = 1;
    cfg.g_timebase = (aom_rational_t){1,50};
    cfg.g_lag_in_frames = 0; cfg.kf_mode = AOM_KF_DISABLED;
    cfg.rc_end_usage = AOM_Q; cfg.rc_min_quantizer = qp; cfg.rc_max_quantizer = qp;
    require(!aom_codec_enc_init(&encoder, iface, &cfg, 0), "init");
    require(!aom_codec_control(&encoder, AOME_SET_CPUUSED, 8), "speed");
    aom_image_t img;
    require(aom_img_alloc(&img, AOM_IMG_FMT_I420, width, height, 1) != NULL, "image");
    fwrite("DKIF",1,4,out); little(out,0,2); little(out,32,2); fwrite("AV01",1,4,out);
    little(out,width,2); little(out,height,2); little(out,50,4); little(out,1,4);
    little(out,count,4); little(out,0,4);
    for (int t = 0; t < count; ++t) {
        for (int p = 0; p < 3; ++p) {
            int w = p ? width/2 : width, h = p ? height/2 : height;
            for (int y = 0; y < h; ++y)
                require(fread(img.planes[p]+y*img.stride[p],1,w,in) == (size_t)w,"short frame");
        }
        aom_svc_ref_frame_config_t refs = {0};
        refs.reference[0] = 1; refs.ref_idx[0] = 0;
        refs.ref_idx[3] = 1;
        refs.reference[3] = (long_ref && t >= 145);
        refs.refresh[0] = 1;
        refs.refresh[1] = t == 99;
        require(!aom_codec_control(&encoder, AV1E_SET_SVC_REF_FRAME_CONFIG, &refs), "refs");
        require(!aom_codec_encode(&encoder,&img,t,1,t==0 ? AOM_EFLAG_FORCE_KF : 0), "encode");
        aom_codec_iter_t it = NULL;
        const aom_codec_cx_pkt_t *pkt;
        int emitted = 0;
        while ((pkt = aom_codec_get_cx_data(&encoder,&it))) {
            if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) continue;
            require(pkt->data.frame.pts == t, "unexpected tick");
            little(out,pkt->data.frame.sz,4); little(out,t,8);
            require(fwrite(pkt->data.frame.buf,1,pkt->data.frame.sz,out)==pkt->data.frame.sz, "payload");
            ++emitted;
        }
        require(emitted == 1,"single low-delay frame required");
    }
    require(fgetc(in) == EOF && !ferror(in), "extra input");
    require(fclose(out) == 0, "flush IVF"); fclose(in);
    aom_img_free(&img); aom_codec_destroy(&encoder);
    return 0;
}
