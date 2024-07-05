from bs4 import BeautifulSoup


def process_files(xml_files, i, cpu):
    output = []
    for n, xml_file in enumerate(xml_files):
        if (n % cpu) == i:
            output.append(process_xml(xml_file))
    return output


def process_xml(xml_file):
    with open(xml_file, 'r') as f:
        data = BeautifulSoup(f.read(), "xml")
    table = data.find('rmsd_table')
    runs = table.find_all("run")
    scores = [float(value.get("binding_energy")) for value in runs]
    with open(data.find("ligand").get_text()) as f:
        for line in f:
            if "Name" in line:
                name = line.split()[-1]
    return name, min(scores)
