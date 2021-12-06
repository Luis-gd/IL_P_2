//Read files
train = []
test = []

for i in range(30)
    //Deporte
    path = "r./dataset/deporte/deporte"+(2*i+1)+".txt"
    path = path.os.split(path)
    with open(path) as f:
         text = f.read()
         train.append([text,0])

    path = "r./dataset/deporte/deporte"+(2*i+2)+".txt"
    path = path.os.split(path)
    with open(path) as f:
         text = f.read()
         test.append([text,0])

    //Salud
    path = "r./dataset/salud/salud"+(2*i+1)+".txt"
    path = path.os.split(path)
    with open(path) as f:
         text = f.read()
         train.append([text,0])

    path = "r./dataset/salud/salud"+(2*i+2)+".txt"
    path = path.os.split(path)
    with open(path) as f:
         text = f.read()
         test.append([text,0])

    //politica
    path = "r./dataset/politica/politica"+(2*i+1)+".txt"
    path = path.os.split(path)
    with open(path) as f:
         text = f.read()
         train.append([text,0])

    path = "r./dataset/politica/politica"+(2*i+2)+".txt"
    path = path.os.split(path)
    with open(path) as f:
         text = f.read()
         test.append([text,0])