import json, os, copy


class Alphabet:
    def __init__(self, name, blankflag=False, padflag=False, unkflag=False, keep_growing=True, path=None):
        self.name = name
        self.PAD = "</pad>"
        self.UNKNOWN = "</unk>"
        self.BLANK = "</blank>"
        self.padflag = padflag
        self.unkflag = unkflag
        self.blankflag = blankflag
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.next_index = 0
        if self.padflag:
            self.add(self.PAD)
            self.pad_id = self.get_index(self.PAD)
        if self.unkflag:
            self.add(self.UNKNOWN)
            self.unk_id = self.get_index(self.UNKNOWN)
        if self.blankflag:
            self.add(self.BLANK)
            self.blank_id = self.get_index(self.BLANK)
        if path:
            self.keep_growing = False
            self.load(path)

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.next_index = 0

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                if self.UNKNOWN in self.instance2index:
                    return self.instance2index[self.UNKNOWN]
                else:
                    print(self.name + " get_index raise wrong, return 0. Please check it")
                    return 0

    def get_instance(self, index):
        if index == 0:
            if self.padflag:
                print(self.name + " get_instance of </pad>, wrong?")
            if not self.padflag and self.unkflag:
                print(self.name + " get_instance of </unk>, wrong?")
            return self.instances[index]
        try:
            return self.instances[index]
        except IndexError:
            print('WARNING: '+ self.name + ' Alphabet get_instance, unknown instance, return the </unk> label.')
            return '</unk>'

    def size(self):
        return len(self.instances)

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def iteritems(self):
        return self.instance2index.items()

    def load(self, path):
        import json
        with open(path) as f:
            vocab = json.load(f)
        for ele in vocab:
            self.add(ele)


