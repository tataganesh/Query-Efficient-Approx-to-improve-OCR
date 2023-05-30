import numpy as np
from apricot import FacilityLocationSelection


def topk(cer_means, num_samples):
    top_k_cers = sorted(cer_means.items(), key=lambda k: k[1], reverse=True)[:num_samples]
    pruned_cers = {name: cer for name, cer in top_k_cers}
    return pruned_cers


def facility_location(cer_means, num_samples):
    cers_array = np.array(list(cer_means.values())).reshape(-1, 1)
    model = FacilityLocationSelection(num_samples, optimizer='naive')
    model.fit(cers_array)
    cer_means_itms = list(cer_means.items())
    pruned_cers = dict()
    counts = 0
    for idx in model.ranking:
        img_name, cer = cer_means_itms[idx]
        pruned_cers[img_name] = cer
        counts += 1
    return pruned_cers
