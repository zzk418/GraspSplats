import sys, os
sys.path.append('./feature-splatting-inria')
import torch
import traceback

from scene import Scene, skip_feat_decoder
from scene.gaussian_model import GaussianModel
import featsplat_editor
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser

parser = ArgumentParser()
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--iteration', default=-1, type=int)
args = parser.parse_args(['-m', 'outputs/tissue_data'])
dataset = lp.extract(args)
opt = op.extract(args)

try:
    print("Loading gaussians...")
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    gaussians.training_setup(opt)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    print(f"Gaussians loaded: {gaussians.get_xyz.shape[0]} points")

    print("Loading feat_decoder...")
    my_feat_decoder = skip_feat_decoder(dataset.distill_feature_dim, part_level=True).cuda()
    decoder_weight_dict = torch.load(os.path.join(dataset.model_path, 'feat_decoder.pth'))
    my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
    my_feat_decoder.eval()
    print("feat_decoder loaded OK")

    print("Loading clip_segmenter...")
    clip_seg = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder)
    print("clip_segmenter loaded OK")

    print("Testing query 'tissue'...")
    sim = clip_seg.compute_similarity_one('tissue', level='object')
    print(f"similarity shape: {sim.shape}, range: [{sim.min().item():.4f}, {sim.max().item():.4f}]")
    print("ALL OK")

except Exception:
    traceback.print_exc()
