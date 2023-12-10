def replace_mask(filename):
    with open(filename, 'r') as file:
        content = file.read()

    content = content.replace('<mask>', '[MASK]')

    content = content.replace(' .', '.')

    with open(filename, 'w') as file:
        file.write(content)

file_path = './experiments/wh_adjunct_mask/input.txt'
replace_mask(file_path)
