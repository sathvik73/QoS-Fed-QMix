import os
import zipfile

def zip_project(output_filename='project_upload.zip'):
    exclude_dirs = {'result', 'model', '__pycache__', '.git', '.idea', 'data'}
    root_dir = os.getcwd()
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Exclude directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file == output_filename:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, arcname)
                
    print(f"Project zipped to {output_filename}")

if __name__ == "__main__":
    zip_project()
