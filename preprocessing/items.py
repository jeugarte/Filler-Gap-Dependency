import sys


def process_text_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as file:
            sentences = file.readlines()

        with open(output_file_path, 'w') as tsv_file:
            global_index = 0
            for sentence in sentences:
                words = sentence.strip().split()
                for local_index, word in enumerate(words):
                    tsv_file.write(f"{global_index}\t{local_index}\t{word}\n")
                    global_index += 1

        return "File processed successfully."
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_text.py input_file.txt output_file.tsv")
    else:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        result = process_text_file(input_file_path, output_file_path)
        print(result)
