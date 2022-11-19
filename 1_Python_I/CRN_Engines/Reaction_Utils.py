##############################################
### author: Abhishek Behera                ###
### email : abhishek.enlightened@gmail.com ###
###         abhishekb@ee.iitb.ac.in        ###
##############################################

import sys,os
from datetime import datetime

if sys.version_info[0] == 2:
    range = xrange

def display(reaction_system):
    for j in range(len(reaction_system[1][0][0])):
        reaction = ''
        for i in range(len(reaction_system[1][0])):
            if reaction_system[1][0][i][j] != 0:
                if reaction_system[1][0][i][j] == 1:
                    reaction += reaction_system[0][i]+' + '
                else:
                    reaction += str(reaction_system[1][0][i][j])+reaction_system[0][i]+' + '
        if reaction[-3:] == ' + ':
            reaction = reaction[:-3]
        else:
            reaction += '0'
        reaction = reaction+' --@rate='+str(reaction_system[1][2][j])+'--> '
        for i in range(len(reaction_system[1][1])):
            if reaction_system[1][1][i][j] != 0:
                if reaction_system[1][1][i][j] == 1:
                    reaction += reaction_system[0][i]+' + '
                else:
                    reaction += str(reaction_system[1][1][i][j])+reaction_system[0][i]+' + '
        if reaction[-3:] == ' + ':
            reaction = reaction[:-3]
        else:
            reaction += '0'
        print(reaction)
    print("\n")
    for s in range(len(reaction_system[0])):
        print(reaction_system[0][s],reaction_system[2][s])

def header(output,translator):
    s = []
    s += ['### Auto-generated with translator: '+translator]
    s += ['### from the model file: '+os.path.basename(output)]
    s += ['### at: '+os.path.dirname(output)]
    d = datetime.now()
    s += ['### (time: '+d.strftime('%Y-%m-%d %H:%M')+')']
    N = max(map(len,s))
    for i in range(len(s)):
        s[i]+=' '*(N-len(s[i]))+' ###\n'
    s = ''.join(s)
    _ = '#'*(N+4)+'\n'
    return _+s+_
                   
def write(reaction_system,output,translator):
    f = open(output,'w')
    f.write(header(output,translator))
    f.write('\nSpecies:\n')
    f.write(','.join(reaction_system[0])+'\n')
    f.write('\nReactions:\n')
    RXN = reaction_system[1]
    for i in range(len(RXN[2])):
        r = ','.join(map(str,RXN[0][:,i]))
        k = '--'+str(RXN[2][i])+'-->'
        p = ','.join(map(str,RXN[1][:,i]))
        f.write(r+k+p+'\n')
    f.write('\n')
    f.write('\nInitial:\n')
    f.write(','.join(map(str,reaction_system[2]))+'\n')
    f.close()
        
