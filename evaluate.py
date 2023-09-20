# References:
    # https://velog.io/@viriditass/GAN%EC%9D%80-%EC%95%8C%EA%B2%A0%EB%8A%94%EB%8D%B0-%EA%B7%B8%EB%9E%98%EC%84%9C-%EC%96%B4%EB%96%A4-GAN%EC%9D%B4-%EB%8D%94-%EC%A2%8B%EC%9D%80%EA%B1%B4%EB%8D%B0-How-to-evaluate-GAN

import torch
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


def calculate_fid(feat1, feat2):
	feat1 = torch.randn(16, 1024)
	feat2 = torch.randn(16, 1024)
	mu1, sigma1 = feat1.mean(axis=0), torch.cov(feat1)
	mu2, sigma2 = feat2.mean(axis=0), torch.cov(feat2)
	sum_squared_diff = torch.sum((mu1 - mu2) ** 2)
	# calculate sqrt of product between cov
    (sigma1 * sigma2) ** 0.5
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = sum_squared_diff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid