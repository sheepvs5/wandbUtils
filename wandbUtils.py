import wandb, torch, os, re

def get_all_runs(project, partial_name=None):
    api = wandb.Api()
    runs = api.runs(project)
    if partial_name is None:
        return runs
    else:
        matched_runs = []
        for run in runs:
            if partial_name in run.name:
                matched_runs.append(run)
        return matched_runs

def cleanup_artifacts(project):
    api = wandb.Api()
    runs = api.runs(project)
    for run in runs:
        artifacts = run.logged_artifacts()
        if len(artifacts)>1:
            for artifact in artifacts:
                if not artifact.aliases:
                    artifact.delete()
                    
def get_files_in_run(run, file_name='script.py'):
    files = []
    for file in run.files():
        if file_name in file.name:
            files.append(file)
    return files

def remove_line_including_word(lines, word):
    line_list = lines.split('\n')
    new_line_list = []        
    for line in line_list:
        if not word in line:
            new_line_list.append(line)
    return '\n'.join(new_line_list)

def remove_line_including_words(lines, words):
    for word in words:
        lines = remove_line_including_word(lines, word)
    return lines

def load_code_from_run(run, remove_keywords=['Repo.clone_from', 'artifact', 'trainer.fit'], file_name='script.py'):
    if isinstance(run, str):
        # api = wandb.Api()
        # run = api.run(run)
        run = get_run_by_name(run)
        
    script_file = get_files_in_run(run, file_name=file_name)[0]
    file = script_file.download('tmp', replace=True)
    with file as f:
        lines = f.read()
    new_lines = remove_line_including_words(lines, remove_keywords)
    return new_lines

def get_last_artifact(artifacts):
    last_version = 0
    last_idx = 0
    for idx, artifact in enumerate(artifacts):
        version = int(re.findall('[0-9]+', artifact.version)[0])
        if version>last_version:
            last_version = version
            last_idx = idx
    return list(artifacts)[last_idx]

def get_run_by_name(name):
    project = name.split('/')[-2]
    runs = wandb.Api().runs(project)
    matched_runs = []
    for run in runs:
        if run.name==name.split('/')[-1]:
            matched_runs.append(run)
    return matched_runs[0]

def get_ckpt_from_artifact(artifact):
    artifact_dir = artifact.download()
    return os.path.join(artifact_dir, 'model.ckpt')

def get_last_model_by_name(name):
    run = get_run_by_name(name)
    artifacts = run.logged_artifacts()
    artifact = get_last_artifact(artifacts)
    return get_ckpt_from_artifact(artifact)
    
def load_model_by_name(name, key=None):
    ''' load state_dict of trained model using name or key.
    If key is not None, then find the model from the key.
    '''
    if key is not None:
        api = wandb.Api()
        run = api.run(key)
    else:
        run = get_run_by_name(name)
    artifacts = run.logged_artifacts()
    artifact = list(artifacts)[-1]
    artifact_dir = artifact.download()
    return torch.load(os.path.join(artifact_dir, 'model.ckpt'), map_location='cpu')['state_dict']
