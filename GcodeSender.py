import sys
import time
import serial

PORT, Baud = 'COM4', 9600
default_input_file = "output.txt"

if sys.argv.__len__() > 1:
    gcodefile = open(sys.argv[1])
else:
    gcodefile = open(default_input_file)

def gcodeCheck(line):
    # check if this is a valid gcode,
    # return false if the code is not valid.
    if line == "":
        print("Invalid G-Code Command:", line)
        return False
    return True

def inputLine(file):
    while True:
        line = file.readline()
        if line == "":
            print("End Of File Reached. Tasks Completed.")
            return False
        line = line.strip()
        if line == "":
            continue
        if gcodeCheck(line):
            return line
        else:
            print("Invalid G-Code Command:", line)
            print("Reading the next Command...")

s = serial.Serial(PORT, Baud)

line = inputLine(gcodefile)

while line:
    print("Command:", line.strip())
    s.write(line.strip().encode())
    # s.read()
    while True:
        if s.readline()=="complete":
            break
    line = inputLine(gcodefile)

gcodefile.close()