import wandb, torch, os, re

def cleanup_artifacts(project):
    api = wandb.Api()
    runs = api.runs(project)
    for run in runs:
        artifacts = run.logged_artifacts()
        if len(artifacts)>1:
            for artifact in artifacts:
                if not artifact.aliases:
                    artifact.delete()
                    
def get_files_by_name(run, name='script.py'):
    files = []
    for file in run.files():
        if name in file.name:
            files.append(file)
    return files

def remove_line_including_word(strings, word):
    lines = strings.split('\n')
    for line in lines:
        if word in line:
            lines.remove(line)
    return '\n'.join(lines)

def remove_lines_including_word(strings, words):
    for word in words:
        strings = remove_line_including_word(strings, word)
    return strings

def load_code_from_run(run_name, remove_keywords=['Repo.clone_from', 'artifact', 'trainer.fit']):
    api = wandb.Api()
    run = api.run(run_name)
    script_file = get_files_by_name(run)[0]
    file = script_file.download('tmp', replace=True)
    with file as f:
        lines = f.read()
    new_lines = remove_lines_including_word(lines, remove_keywords)
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

def get_run_by_name(project, name):
    runs = wandb.Api().runs(project)
    matched_runs = []
    for run in runs:
        if run.name==name:
            matched_runs.append(run)
    return matched_runs[0]

def get_ckpt_from_artifact(artifact):
    artifact_dir = artifact.download()
    return os.path.join(artifact_dir, 'model.ckpt')

def get_last_model_by_name(project, name):
    run = get_run_by_name(project, name)
    artifacts = run.logged_artifacts()
    artifact = get_last_artifact(artifacts)
    return get_ckpt_from_artifact(artifact)
    
def load_model_from_run(project, name, key=None):
    ''' load state_dict of trained model using name or key.
    If key is not None, then find the model from the key.
    '''
    if key is not None:
        api = wandb.Api()
        run = api.run(run_name)
    else:
        run = get_run_by_name(project, name)
    artifacts = run.logged_artifacts()
    artifact = list(artifacts)[-1]
    artifact_dir = artifact.download()
    return torch.load(os.path.join(artifact_dir, 'model.ckpt'))['state_dict']
