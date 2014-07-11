package org.saliya.ompi.util;

import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;


public class MpiOps {
    public static String allReduceStr(String value, Intracomm comm) throws MPIException {
        int size = comm.getSize();
        int [] lengths = new int[size];
        int length = value.length();
        int rank = comm.getRank();
        lengths[rank] = length;
        comm.allGather(lengths, 1, MPI.INT);
        int [] displas = new int[size];
        displas[0] = 0;
        System.arraycopy(lengths, 0, displas, 1, size - 1);
        if (displas.length > 2){
            for (int i = 2; i < displas.length; ++i){
                displas[i] += displas[i-1];
            }
        }

        /*Arrays.parallelPrefix(displas, new IntBinaryOperator() {
            @Override
            public int applyAsInt(int m, int n) {
                return m + n;
            }
        });*/
        int count = 0;
        for (int i=0; i < lengths.length; ++i){
            count += lengths[i];
        }
        char [] recv = new char[count];
        System.arraycopy(value.toCharArray(), 0,recv, displas[rank], length);
        comm.allGatherv(recv, lengths, displas, MPI.CHAR);
        return  new String(recv);
    }
}
