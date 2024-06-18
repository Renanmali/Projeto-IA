import os

def rename_files(directory):
    files = os.listdir(directory)
    files.sort()  # Ordena os arquivos para garantir a sequência correta
    for i, filename in enumerate(files):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"ffhq_{i+1}{file_extension}"
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")

directory = 'neutral'  # Substitua pelo caminho do seu diretório
rename_files(directory)