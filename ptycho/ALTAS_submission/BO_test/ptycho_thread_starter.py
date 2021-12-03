import sys
import os

# TODO: also prepare the slurm script using this python script, then run the .sub file with initialized files.
# Currently, the PARFILELIST in the slurm script needs to be manually changed to match the parameter files created here.
def main(idx: int):
    template = 'ptycho_thread_template.sub'
    filename = 'ptycho_thread' + str(idx) + '.sub'
    if os.path.isfile(filename):
        os.remove(filename)
    with open(template) as f:
        filedata = f.read()
    f.close()
    filedata = filedata.replace('&THREAD_NUM&', str(idx))
    with open(filename, "x") as f:
        f.write(filedata)
    f.close()
    cmd = 'sbatch ' + filename
    try:
        os.system(cmd)
    except:
        print("Job submision failed.")
    finally:
        return

if __name__ == "__main__":
    main(sys.argv[1])