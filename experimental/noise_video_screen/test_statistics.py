import numpy as np
from .statistics import assess,merge_stats


def test_merge_weights_valid_pixels_and_avoids_cross_window_temporal_edges():
    a=dict(sse=2.,sae=2.,count=2,unseen_sse=1.,unseen_count=1,observed_sse=1.,observed_count=1,frame_psnr_sum=20.,frames=2,temporal_sse=3.,temporal_count=1)
    b=dict(a,sse=6.,count=6,unseen_sse=9.,unseen_count=3,temporal_sse=5.,temporal_count=1)
    m=merge_stats([a,b]);assert m['unobserved_mse']==2.5
    assert m['mse']==1 and m['psnr']==0 and m['temporal_delta_mse']==4


def test_diagonal_count_and_gaps():
    m=np.ones((4,4))*3;np.fill_diagonal(m,1)
    assert assess(m)['pass_count']==4
    m[1,0]=.5
    d=assess(m);assert d['pass_count']==3 and d['row_min_input'][1]==0
    assert d['relative_gap'][1]==1


def test_ties_allowed_only_at_numerical_precision():
    m=np.ones((4,4));assert assess(m)['all_four']
    m[2,2]+=1e-8;assert assess(m)['pass_count']==3
