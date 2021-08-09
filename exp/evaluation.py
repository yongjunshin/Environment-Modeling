import torch
import numpy as np

from src.soft_dtw_cuda import SoftDTW


def simulate_deterministic(env_model, sut, sim_length, initial_state_batch, device):
    sim_result = torch.zeros((initial_state_batch.shape[0], sim_length, initial_state_batch.shape[2]),
                             device=device)
    sim_x = initial_state_batch

    env_model.eval()
    for sim_idx in range(sim_length):
        # action choice
        action = env_model(sim_x).detach()

        # state transition
        sys_operations = sut.act_sequential(action.cpu().numpy())
        sys_operations = torch.tensor(sys_operations).to(device=device).type(torch.float32)
        next_x = torch.cat((action, sys_operations), dim=1)
        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
        sim_x = sim_x[:, 1:]
        sim_x = torch.cat((sim_x, next_x), dim=1)
        sim_result[:, sim_idx] = sim_x[:, -1]

    return sim_result


def simulate_non_deterministic(env_model, sut, sim_length, initial_state_batch, device):
    sim_result = torch.zeros((initial_state_batch.shape[0], sim_length, initial_state_batch.shape[2]),
                             device=device)
    sim_x = initial_state_batch

    env_model.eval()
    for sim_idx in range(sim_length):
        # action choice
        action_prob = env_model.get_distribution(sim_x)
        action = action_prob.sample().detach()

        # state transition
        sys_operations = sut.act_sequential(action.cpu().numpy())
        sys_operations = torch.tensor(sys_operations).to(device=device).type(torch.float32)
        next_x = torch.cat((action, sys_operations), dim=1)
        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
        sim_x = sim_x[:, 1:]
        sim_x = torch.cat((sim_x, next_x), dim=1)
        sim_result[:, sim_idx] = sim_x[:, -1]

    return sim_result


def batch_euclidean_distance(batch1, batch2):
    diffs = torch.pow(batch1 - batch2, 2)
    diffs = torch.sum(diffs, dim=(1, 2))
    diffs = torch.sqrt(diffs)
    return diffs


sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1)


def batch_dynamic_time_warping(batch1, batch2):
    diffs = sdtw(batch1, batch2)
    return diffs


gray_underbound = 35
gray_upperbound = 49

def batch_time_length_on_line_border_comparison(batch1, batch2, normalizer, device):
    unnormalized_underbound = gray_underbound
    normalized_underbound = normalizer.transform(np.array([[unnormalized_underbound, 0]]))[0, 0]
    normalized_underbound = torch.tensor([normalized_underbound], device=device)

    unnormalized_upperbound = gray_upperbound
    normalized_upperbound = normalizer.transform(np.array([[unnormalized_upperbound, 0]]))[0, 0]
    normalized_upperbound = torch.tensor([normalized_upperbound], device=device)

    true_mask_value = torch.tensor([1.], device=device)
    false_mask_value = torch.tensor([0.], device=device)

    batch1_undergray_mask = masking_s(batch1, normalized_underbound, true_mask_value, false_mask_value)
    batch1_uppergray_mask = masking_g(batch1, normalized_upperbound, true_mask_value, false_mask_value)
    batch2_undergray_mask = masking_s(batch2, normalized_underbound, true_mask_value, false_mask_value)
    batch2_uppergray_mask = masking_g(batch2, normalized_upperbound, true_mask_value, false_mask_value)

    batch1_gray_geq_1tick_mask = -(batch1_undergray_mask + batch1_uppergray_mask) + 1
    batch2_gray_geq_1tick_mask = -(batch2_undergray_mask + batch2_uppergray_mask) + 1

    batch1_gray_0tick_mask = batch1_undergray_mask[:, :-1] * batch1_uppergray_mask[:, 1:] + batch1_undergray_mask[:, 1:] * batch1_uppergray_mask[:, :-1]
    batch2_gray_0tick_mask = batch2_undergray_mask[:, :-1] * batch2_uppergray_mask[:, 1:] + batch2_undergray_mask[:, 1:] * batch2_uppergray_mask[:, :-1]

    batch1_length_mask = length_masking(batch1_gray_geq_1tick_mask, device)
    batch2_length_mask = length_masking(batch2_gray_geq_1tick_mask, device)

    one = torch.tensor([1.], device=device)
    batch1_length_mask_mask = masking_geq(batch1_length_mask, one, true_mask_value, false_mask_value)
    batch2_length_mask_mask = masking_geq(batch2_length_mask, one, true_mask_value, false_mask_value)

    length_sum_batch1 = torch.sum(batch1_length_mask, dim=(1, 2))
    length_sum_batch2 = torch.sum(batch2_length_mask, dim=(1, 2))

    count_batch1 = torch.sum(batch1_length_mask_mask, dim=(1, 2)) + torch.sum(batch1_gray_0tick_mask, dim=(1, 2))
    count_batch2 = torch.sum(batch2_length_mask_mask, dim=(1, 2)) + torch.sum(batch2_gray_0tick_mask, dim=(1, 2))

    mean_length_batch1 = length_sum_batch1 / count_batch1
    mean_length_batch2 = length_sum_batch2 / count_batch2
    mean_diffs = torch.abs(mean_length_batch1 - mean_length_batch2)
    # l = 50
    # print(torch.cat((batch1[0][:l], batch1_undergray_mask[0][:l], batch1_uppergray_mask[0][:l], batch1_gray_geq_1tick_mask[0][:l], batch1_length_mask[0][:l], batch1_gray_0tick_mask[0][:l]), dim=1))


    batch1_length_geq_1tick_hist = batch_histogram(batch1_length_mask, count_batch1, 0, 400, 400, device)
    batch2_length_geq_1tick_hist = batch_histogram(batch2_length_mask, count_batch2, 0, 400, 400, device)

    batch1_length_0tick_hist = torch.sum(batch1_gray_0tick_mask, dim=(1, 2)) / count_batch1
    batch2_length_0tick_hist = torch.sum(batch2_gray_0tick_mask, dim=(1, 2)) / count_batch2
    batch1_length_0tick_hist = torch.reshape(batch1_length_0tick_hist, (len(batch1_length_0tick_hist), 1, 1))
    batch2_length_0tick_hist = torch.reshape(batch2_length_0tick_hist, (len(batch2_length_0tick_hist), 1, 1))

    batch1_length_hist = torch.cat((batch1_length_0tick_hist, batch1_length_geq_1tick_hist), dim=1)
    batch2_length_hist = torch.cat((batch2_length_0tick_hist, batch2_length_geq_1tick_hist), dim=1)
    check_sum1 = torch.sum(batch1_length_hist, dim=(1, 2))
    check_sum2 = torch.sum(batch2_length_hist, dim=(1, 2))

    #print(batch1_length_hist[0])
    #
    jsd_diff = batch_JSD(batch1_length_hist, batch2_length_hist, device)

    mean_diffs = avoid_nan_to_avg(mean_diffs, device)
    jsd_diff = avoid_nan_to_avg(jsd_diff, device)
    return mean_diffs, jsd_diff


def batch_time_length_outside_line_border_comparison(batch1, batch2, normalizer, device):
    unnormalized_underbound = gray_underbound
    normalized_underbound = normalizer.transform(np.array([[unnormalized_underbound, 0]]))[0, 0]
    normalized_underbound = torch.tensor([normalized_underbound], device=device)

    unnormalized_upperbound = gray_upperbound
    normalized_upperbound = normalizer.transform(np.array([[unnormalized_upperbound, 0]]))[0, 0]
    normalized_upperbound = torch.tensor([normalized_upperbound], device=device)

    true_mask_value = torch.tensor([1.], device=device)
    false_mask_value = torch.tensor([0.], device=device)

    batch1_mask1 = masking_s(batch1, normalized_underbound, true_mask_value, false_mask_value)
    batch1_mask2 = masking_g(batch1, normalized_upperbound, true_mask_value, false_mask_value)

    batch2_mask1 = masking_s(batch2, normalized_underbound, true_mask_value, false_mask_value)
    batch2_mask2 = masking_g(batch2, normalized_upperbound, true_mask_value, false_mask_value)

    batch1_length_mask1 = length_masking(batch1_mask1, device)
    batch1_length_mask2 = length_masking(batch1_mask2, device)
    batch2_length_mask1 = length_masking(batch2_mask1, device)
    batch2_length_mask2 = length_masking(batch2_mask2, device)

    # batch1_length_mask = batch1_length_mask1 + batch1_length_mask2
    # batch2_length_mask = batch2_length_mask1 + batch2_length_mask2

    one = torch.tensor([1.], device=device)
    batch1_length_mask1_mask = masking_geq(batch1_length_mask1, one, true_mask_value, false_mask_value)
    batch1_length_mask2_mask = masking_geq(batch1_length_mask2, one, true_mask_value, false_mask_value)
    batch2_length_mask1_mask = masking_geq(batch2_length_mask1, one, true_mask_value, false_mask_value)
    batch2_length_mask2_mask = masking_geq(batch2_length_mask2, one, true_mask_value, false_mask_value)

    # print(torch.cat((batch1[0], batch1_mask1[0], batch1_mask2[0], batch1_length_mask1[0], batch1_length_mask2[0], batch1_length_mask[0], batch1_length_mask_mask[0]), dim=1)[:50])
    # print(torch.cat((batch2[0], batch2_mask1[0], batch2_mask2[0], batch2_length_mask1[0], batch2_length_mask2[0], batch2_length_mask[0], batch2_length_mask_mask[0]), dim=1)[:50])

    length_sum1_batch1 = torch.sum(batch1_length_mask1, dim=(1, 2))
    length_sum2_batch1 = torch.sum(batch1_length_mask2, dim=(1, 2))
    length_sum1_batch2 = torch.sum(batch2_length_mask1, dim=(1, 2))
    length_sum2_batch2 = torch.sum(batch2_length_mask2, dim=(1, 2))

    count1_batch1 = torch.sum(batch1_length_mask1_mask, dim=(1, 2))
    count2_batch1 = torch.sum(batch1_length_mask2_mask, dim=(1, 2))
    count1_batch2 = torch.sum(batch2_length_mask1_mask, dim=(1, 2))
    count2_batch2 = torch.sum(batch2_length_mask2_mask, dim=(1, 2))

    mean_length1_batch1 = length_sum1_batch1 / count1_batch1
    mean_length2_batch1 = length_sum2_batch1 / count2_batch1
    mean_length1_batch2 = length_sum1_batch2 / count1_batch2
    mean_length2_batch2 = length_sum2_batch2 / count2_batch2

    mean_diffs1 = torch.abs(mean_length1_batch1 - mean_length1_batch2)
    mean_diffs2 = torch.abs(mean_length2_batch1 - mean_length2_batch2)
    mean_diffs = 0.5 * (mean_diffs1 + mean_diffs2)

    batch1_length_hist1 = batch_histogram(batch1_length_mask1, count1_batch1, 0, 400, 400, device)
    batch1_length_hist2 = batch_histogram(batch1_length_mask2, count2_batch1, 0, 400, 400, device)
    batch2_length_hist1 = batch_histogram(batch2_length_mask1, count1_batch2, 0, 400, 400, device)
    batch2_length_hist2 = batch_histogram(batch2_length_mask2, count2_batch2, 0, 400, 400, device)
    check_sum11 = torch.sum(batch1_length_hist1, dim=(1, 2))
    check_sum12 = torch.sum(batch1_length_hist2, dim=(1, 2))
    check_sum21 = torch.sum(batch2_length_hist1, dim=(1, 2))
    check_sum22 = torch.sum(batch2_length_hist2, dim=(1, 2))

    jsd_diff1 = batch_JSD(batch1_length_hist1, batch2_length_hist1, device)
    jsd_diff2 = batch_JSD(batch1_length_hist2, batch2_length_hist2, device)
    jsd_diff = 0.5 * (jsd_diff1 + jsd_diff2)

    mean_diffs = avoid_nan_to_avg(mean_diffs, device)
    jsd_diff = avoid_nan_to_avg(jsd_diff, device)
    return mean_diffs, jsd_diff


def batch_amplitude_comparison(batch1, batch2, normalizer, device):
    unnormalized_underbound = gray_underbound
    normalized_underbound = normalizer.transform(np.array([[unnormalized_underbound, 0]]))[0, 0]
    normalized_underbound = torch.tensor([normalized_underbound], device=device)

    unnormalized_upperbound = gray_upperbound
    normalized_upperbound = normalizer.transform(np.array([[unnormalized_upperbound, 0]]))[0, 0]
    normalized_upperbound = torch.tensor([normalized_upperbound], device=device)

    true_mask_value = torch.tensor([1.], device=device)
    false_mask_value = torch.tensor([0.], device=device)
    zero = torch.tensor([0.], device=device)

    batch1_mask1 = masking_s(batch1, normalized_underbound, true_mask_value, false_mask_value)
    batch1_mask2 = masking_g(batch1, normalized_upperbound, true_mask_value, false_mask_value)

    batch2_mask1 = masking_s(batch2, normalized_underbound, true_mask_value, false_mask_value)
    batch2_mask2 = masking_g(batch2, normalized_upperbound, true_mask_value, false_mask_value)

    batch1_amplitude1 = (-1 * (batch1 - normalized_underbound)) * batch1_mask1
    batch1_amplitude2 = (batch1 - normalized_upperbound) * batch1_mask2

    batch2_amplitude1 = (-1 * (batch2 - normalized_underbound)) * batch2_mask1
    batch2_amplitude2 = (batch2 - normalized_upperbound) * batch2_mask2

    #test
    #test_batch = torch.tensor([[[0.], [0.0976], [0.4634], [0.5610], [0.5366], [0.5366], [0.4390], [0.3171], [0.2683], [0.]]], device=device)
    #test_mas = max_amplitude_masking(test_batch, device)

    batch1_amplitude1 = max_amplitude(batch1_amplitude1, device)
    batch1_amplitude2 = max_amplitude(batch1_amplitude2, device)
    #batch1_amplitude = batch1_amplitude1 + batch1_amplitude2
    batch1_amplitude1_mask = masking_g(batch1_amplitude1, zero, true_mask_value, false_mask_value)
    batch1_amplitude2_mask = masking_g(batch1_amplitude2, zero, true_mask_value, false_mask_value)

    batch2_amplitude1 = max_amplitude(batch2_amplitude1, device)
    batch2_amplitude2 = max_amplitude(batch2_amplitude2, device)
    batch2_amplitude1_mask = masking_g(batch2_amplitude1, zero, true_mask_value, false_mask_value)
    batch2_amplitude2_mask = masking_g(batch2_amplitude2, zero, true_mask_value, false_mask_value)

    amplitude_sum1_batch1 = torch.sum(batch1_amplitude1, dim=(1, 2))
    amplitude_sum2_batch1 = torch.sum(batch1_amplitude2, dim=(1, 2))
    amplitude_sum1_batch2 = torch.sum(batch2_amplitude1, dim=(1, 2))
    amplitude_sum2_batch2 = torch.sum(batch2_amplitude2, dim=(1, 2))

    count1_batch1 = torch.sum(batch1_amplitude1_mask, dim=(1, 2))
    count2_batch1 = torch.sum(batch1_amplitude2_mask, dim=(1, 2))
    count1_batch2 = torch.sum(batch2_amplitude1_mask, dim=(1, 2))
    count2_batch2 = torch.sum(batch2_amplitude2_mask, dim=(1, 2))

    mean_amplitude1_batch1 = amplitude_sum1_batch1 / count1_batch1
    mean_amplitude2_batch1 = amplitude_sum2_batch1 / count2_batch1
    mean_amplitude1_batch2 = amplitude_sum1_batch2 / count1_batch2
    mean_amplitude2_batch2 = amplitude_sum2_batch2 / count2_batch2

    mean_diffs1 = torch.abs(mean_amplitude1_batch1 - mean_amplitude1_batch2)
    mean_diffs2 = torch.abs(mean_amplitude2_batch1 - mean_amplitude2_batch2)
    mean_diffs = 0.5 * (mean_diffs1 + mean_diffs2)

    batch1_amplitude_hist1 = batch_histogram(batch1_amplitude1, count1_batch1, 0, 2, 40, device)
    batch1_amplitude_hist2 = batch_histogram(batch1_amplitude2, count2_batch1, 0, 2, 40, device)
    batch2_amplitude_hist1 = batch_histogram(batch2_amplitude1, count1_batch2, 0, 2, 40, device)
    batch2_amplitude_hist2 = batch_histogram(batch2_amplitude2, count2_batch2, 0, 2, 40, device)
    check_sum11 = torch.sum(batch1_amplitude_hist1, dim=(1, 2))
    check_sum12 = torch.sum(batch1_amplitude_hist2, dim=(1, 2))
    check_sum21 = torch.sum(batch2_amplitude_hist1, dim=(1, 2))
    check_sum22 = torch.sum(batch2_amplitude_hist2, dim=(1, 2))

    jsd_diff1 = batch_JSD(batch1_amplitude_hist1, batch2_amplitude_hist1, device)
    jsd_diff2 = batch_JSD(batch1_amplitude_hist2, batch2_amplitude_hist2, device)
    jsd_diff = 0.5 * (jsd_diff1 + jsd_diff2)

    mean_diffs = avoid_nan_to_avg(mean_diffs, device)
    jsd_diff = avoid_nan_to_avg(jsd_diff, device)
    return mean_diffs, jsd_diff


def masking_g_seq(batch, under_bound, upper_bound, true_mask_value, false_mask_value):
    under_mask = torch.where(batch > under_bound, true_mask_value, false_mask_value)
    upper_mask = torch.where(batch <= upper_bound, true_mask_value, false_mask_value)
    mask = under_mask * upper_mask
    return mask


def masking_geq(batch, under_bound, true_mask_value, false_mask_value):
    under_mask = torch.where(batch >= under_bound, true_mask_value, false_mask_value)
    return under_mask


def masking_g(batch, under_bound, true_mask_value, false_mask_value):
    under_mask = torch.where(batch > under_bound, true_mask_value, false_mask_value)
    return under_mask


def masking_s(batch, upper_bound, true_mask_value, false_mask_value):
    upper_mask = torch.where(batch < upper_bound, true_mask_value, false_mask_value)
    return upper_mask


def avoid_nan_to_avg(batch, device):
    if (not torch.isnan(batch).any()) or torch.isnan(batch).all():
        return batch
    else:
        nan_index = torch.isnan(batch)

        nan_to_zero_batch = torch.nan_to_num(batch)
        avg = torch.sum(nan_to_zero_batch) / torch.sum(~nan_index)
        avg_batch = avg.repeat(len(batch))

        nan_to_avg_batch = ~nan_index * nan_to_zero_batch + nan_index * avg_batch
        return nan_to_avg_batch


def length_masking(batch, device):
    length_mask = torch.zeros_like(batch, device=device)
    for sim_idx in range(1, batch.shape[1]):
        length_mask[:, sim_idx] = (batch[:, sim_idx] + length_mask[:, sim_idx-1]) * batch[:, sim_idx]
        negate = torch.abs(batch[:, sim_idx] - 1)
        length_mask[:, sim_idx-1] = negate * length_mask[:, sim_idx-1]
    return length_mask


def max_amplitude(batch, device):
    zero = torch.tensor([0.], device=device)
    zeros = torch.zeros_like(batch[:, 0], device=device)
    true_mask_value = torch.tensor([1.], device=device)
    false_mask_value = torch.tensor([0.], device=device)

    max_amplitude = torch.zeros_like(batch, device=device)
    max_amplitude[:, 0] = torch.where(batch[:, 0] > zero, true_mask_value, false_mask_value) * batch[:, 0]
    for sim_idx in range(1, batch.shape[1]):
        zero_check = torch.where(batch[:, sim_idx] <= zero, true_mask_value, false_mask_value)
        positive_check = torch.where(batch[:, sim_idx] > zero, true_mask_value, false_mask_value)
        ge_check = torch.where(batch[:, sim_idx] >= max_amplitude[:, sim_idx-1], true_mask_value, false_mask_value)
        smaller_check = torch.where(batch[:, sim_idx] < max_amplitude[:, sim_idx-1], true_mask_value, false_mask_value)

        max_amplitude[:, sim_idx] = ((positive_check * ge_check) * batch[:, sim_idx])\
                                    + ((positive_check * smaller_check) * max_amplitude[:, sim_idx-1])

        max_amplitude[:, sim_idx-1] = ((positive_check * ge_check) * zeros)\
                                    + ((positive_check * smaller_check) * zeros)\
                                    + (zero_check * max_amplitude[:, sim_idx-1])
        #print(torch.cat((batch[0], max_amplitude[0]), dim=1))
    return max_amplitude


def batch_histogram(batch, batch_count, min, max, num_item, device):
    batch_hist = torch.zeros((batch.shape[0], num_item, 1), device=device)
    edges = []
    edges.append(min)
    diff = (max - min) / num_item
    for i in range(1, num_item + 1):
        edges.append(edges[i-1] + diff)

    true_mask_value = torch.tensor([1.], device=device)
    false_mask_value = torch.tensor([0.], device=device)
    for i in range(num_item):
        under_edge = edges[i]
        upper_edge = edges[i+1]
        counts = masking_g_seq(batch, under_edge, upper_edge, true_mask_value, false_mask_value)
        counts = torch.sum(counts, dim=(1, 2))
        batch_hist[:, i] = torch.reshape((counts / batch_count), (batch.shape[0], 1))

    return batch_hist


def batch_KLD(target_batch, ref_batch, device):
    batch_kld = torch.zeros_like(target_batch)

    true_mask_value = torch.tensor([1.], device=device)
    false_mask_value = torch.tensor([0.], device=device)
    zero = torch.tensor([0.], device=device)
    zeros = torch.zeros_like(target_batch[:, 0], device=device)

    magic_number_for_div0 = 0.0000001
    for i in range(target_batch.shape[1]):
        ref_zero_check = torch.where(target_batch[:, i] <= zero, true_mask_value, false_mask_value)
        ref_positive_check = torch.where(target_batch[:, i] > zero, true_mask_value, false_mask_value)

        kld_value = torch.log(target_batch[:, i] / (ref_batch[:, i] + magic_number_for_div0))
        kld_value = torch.nan_to_num(kld_value)
        kld_value = kld_value * target_batch[:, i]
        batch_kld[:, i] = ref_zero_check * zeros + ref_positive_check * kld_value

    batch_kld = torch.sum(batch_kld, dim=(1, 2))
    return batch_kld


def batch_JSD(batch1, batch2, device):
    average_hist = (batch1 + batch2) / 2
    batch_jsd = 0.5 * batch_KLD(batch1, average_hist, device) + 0.5 * batch_KLD(batch2, average_hist, device)
    return batch_jsd


def simulation_and_comparison(model, sut, testing_dataloader, device):
    num_data = len(testing_dataloader.dataset)

    ed_sum = torch.zeros((), device=device)
    dtw_sum = torch.zeros((), device=device)
    time_on_border_diff_sum = torch.zeros((), device=device)
    time_on_border_jsd_diff_sum = torch.zeros((), device=device)

    time_outside_border_diff_sum = torch.zeros((), device=device)
    time_outside_border_jsd_diff_sum = torch.zeros((), device=device)

    amplitude_diff_sum = torch.zeros((), device=device)
    amplitude_jsd_diff_sum = torch.zeros((), device=device)

    for _, (x_batch, y_batch) in enumerate(testing_dataloader):
        sim_result = simulate_deterministic(model, sut, y_batch.shape[1], x_batch, device)
        euclidean_distances = batch_euclidean_distance(sim_result[:, :, [0]], y_batch[:, :, [0]])
        dtws = batch_dynamic_time_warping(sim_result[:, :, [0]], y_batch[:, :, [0]])
        mean_diff_of_time_on_border, jsd_diff_of_time_on_border = batch_time_length_on_line_border_comparison(sim_result[:, :, [0]], y_batch[:, :, [0]], sut.get_normalizer(), device)
        mean_diff_of_time_outside_border, jsd_diff_of_time_outside_border = batch_time_length_outside_line_border_comparison(sim_result[:, :, [0]], y_batch[:, :, [0]], sut.get_normalizer(), device)
        mean_diff_of_amplitude, jsd_diff_of_amplitude = batch_amplitude_comparison(sim_result[:, :, [0]], y_batch[:, :, [0]], sut.get_normalizer(), device)

        ed_sum = ed_sum + torch.sum(euclidean_distances)
        dtw_sum = dtw_sum + torch.sum(dtws)

        time_on_border_diff_sum = time_on_border_diff_sum + torch.sum(mean_diff_of_time_on_border)
        time_on_border_jsd_diff_sum = time_on_border_jsd_diff_sum + torch.sum(jsd_diff_of_time_on_border)

        time_outside_border_diff_sum = time_outside_border_diff_sum + torch.sum(mean_diff_of_time_outside_border)
        time_outside_border_jsd_diff_sum = time_outside_border_jsd_diff_sum + torch.sum(jsd_diff_of_time_outside_border)

        amplitude_diff_sum = amplitude_diff_sum + torch.sum(mean_diff_of_amplitude)
        amplitude_jsd_diff_sum = amplitude_jsd_diff_sum + torch.sum(jsd_diff_of_amplitude)

    ed_mean = ed_sum / num_data
    dtw_mean = dtw_sum / num_data

    time_on_border_diff_mean = time_on_border_diff_sum / num_data
    time_on_border_jsd_diff_mean = time_on_border_jsd_diff_sum / num_data

    time_outside_border_diff_mean = time_outside_border_diff_sum / num_data
    time_outside_border_jsd_diff_mean = time_outside_border_jsd_diff_sum / num_data

    amplitude_diff_mean = amplitude_diff_sum / num_data
    amplitude_jsd_diff_mean = amplitude_jsd_diff_sum / num_data

    report = [ed_mean.item(), dtw_mean.item(), time_on_border_diff_mean.item(), time_outside_border_diff_mean.item(), amplitude_diff_mean.item(), time_on_border_jsd_diff_mean.item(), time_outside_border_jsd_diff_mean.item(), amplitude_jsd_diff_mean.item()]
    print(report)
    return report   # [ed_mean, dtw_mean, time_on_border_diff_mean, time_outside_border_diff_mean, amplitude_diff_mean, time_on_border_jsd_diff_mean, time_outside_border_jsd_diff_mean, amplitude_jsd_diff_mean]
