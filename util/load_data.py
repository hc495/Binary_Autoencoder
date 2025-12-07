import random

def make_test_samples_from_experimentor(
    experimentor,
    k,
    sample_number = 64,
    sample_random_seed = 42
):
    original_set = experimentor.calibration_set()
    prompt_writter = experimentor.prompt_former

    ## Sample k+1 inputs
    random.seed(sample_random_seed)
    sampled_indics = []
    for i in range(sample_number):
        sampled_indics.append(random.sample(range(len(original_set)), (k+1)))
    
    ## Find label sets
    labels = []
    for samples in sampled_indics:
        label = []
        for index in samples:
            label_word = original_set[index][1]
            label_index = prompt_writter._label_space.index(label_word)
            label.append(label_index)
        labels.append(label)
    
    ## Form the prompts
    prompts = []
    for i in range(sample_number):
        demo_lines = []
        for index in range(len(sampled_indics[i]) - 1):
            demo_lines.append(original_set[index])
        query_line = original_set[sampled_indics[i][-1]][0]
        prompt = prompt_writter.write_prompt_from_dataline(demo_lines, query_line)
        if prompt_writter._label_prefix[-1] == ' ':
            prompt = prompt[:-1]
        prompts.append(prompt)
    
    return prompts, labels