import os, re
from git.repo import Repo
from git import cmd
import csv
from itertools import islice
import prettytable

class version_controller:
    repo: Repo
    history: list
    cur_version: int
    dir: str

    def __init__(self, directory: str):
        self.dir = directory
        if os.path.exists(directory + "\\.git") and \
                os.path.exists(directory + "\\.history") and \
                os.path.exists(directory + "\\.gitignore"):
            self.repo = Repo(directory)
            self.git = self.repo.git
            history_file = open(directory + "\\.history", "r", encoding = "UTF-8")
            reader = csv.reader(history_file)
            self.history = []
            self.cur_version = 0
            for ln in islice(reader, 1, None):
                self.history.append(ln)
                self.cur_version += 1
            history_file.close()

        elif (not os.path.exists(directory + "\\.git")) and \
                (not os.path.exists(directory + "\\.history")) and \
                (not os.path.exists(directory + "\\.gitignore")):
            self.repo = Repo.init(directory)
            self.git = self.repo.git
            ignore_file = open(directory + "\\.gitignore", 'w', encoding="UTF-8", newline="")
            ignore_file.write("data/\n.history\n.gitignore\n")
            ignore_file.close()
            self.history = []
            history_file = open(directory + "\\.history", 'w', encoding="UTF-8", newline="")
            history_file.write("version,version_code,training_loss,testing_loss,training_time,extra_message")
            history_file.close()
            self.cur_version = 0
        else:
            raise Exception("directory damage")

    # save the current version code and config files
    def save(self):
        self.git.add(".")
        self.cur_version += 1
        try:
            commit_msg = self.git.commit('-m', "{}".format(self.cur_version))
        except cmd.GitCommandError:
            return
        version_code = re.findall(r"(.......)]", commit_msg)[0]
        self.history.append(['{}'.format(self.cur_version),'{}'.format(version_code),"NULL","NULL","NULL",""])
        history_file = open(self.dir + "\\.history", 'w', encoding="UTF-8", newline="")
        history_file.write("version,version_code,training_loss,testing_loss,training_time,extra_message\n")
        for his in self.history:
            for i in range(6):
                history_file.write(his[i])
                if i != 5:
                    history_file.write(",")
            history_file.write('\n')
        history_file.close()
        return len(self.history) # return the position of current record

    # update experiment result of current version
    def update_result(self, version, training_loss: str = "NULL", testing_loss: str = "NULL", training_time: str = "NULL", extra_message: str = ""):
        if version > len(self.history) or version <= 0:
            raise Exception("version {} does not exist!".format(version))
        self.history[version][2] = str(training_loss)
        self.history[version][3] = str(testing_loss)
        self.history[version][4] = str(training_time)
        self.history[version][5] = extra_message
        history_file = open(self.dir + "\\.history", 'w', encoding="UTF-8", newline="")
        history_file.write("version,version_code,training_loss,testing_loss,training_time,extra_message\n")
        for his in self.history:
            for i in range(6):
                history_file.write(his[i])
                if i != 5:
                    history_file.write(",")
            history_file.write('\n')
        history_file.close()
    def recover(self, version: int):
        if version > len(self.history) or version <= 0:
            raise Exception("version {} does not exist!".format(version))
        version_code = self.history[version-1][1]
        self.git.reset(version_code)
        self.git.clean("-f")
        self.git.stash()
        self.cur_version += 1
        self.history.append(['{}'.format(self.cur_version),
                             '{}'.format(version_code),
                             '{}'.format("NULL"),
                             '{}'.format("NULL"),
                             '{}'.format("NULL"),
                             'recover from version {}'.format(version)])
        history_file = open(self.dir + "\\.history", 'w', encoding="UTF-8", newline="")
        history_file.write("version,version_code,training_loss,testing_loss,training_time,extra_message\n")
        for his in self.history:
            for i in range(6):
                history_file.write(his[i])
                if i != 5:
                    history_file.write(",")
            history_file.write('\n')
        history_file.close()

    def get_history(self):
        return self.history

    def print_history(self):
        table = prettytable.PrettyTable(["version","training_loss","testing_loss","training_time","extra_message"])
        for his in self.history:
            asst = [his[0], his[2], his[3], his[4], his[5]]
            table.add_row(asst)
        print(table)
