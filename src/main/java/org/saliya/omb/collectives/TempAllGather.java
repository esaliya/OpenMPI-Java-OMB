package org.saliya.omb.collectives;

import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;

import java.nio.ByteBuffer;

public class TempAllGather {
    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        Intracomm comm = MPI.COMM_WORLD;
        int rank = comm.getRank();
        int numProcs = comm.getSize();

        int maxMsgSize = Integer.parseInt(args[0]);
		System.out.println(maxMsgSize);
        int byteBytes = maxMsgSize;
        ByteBuffer sbuff = MPI.newByteBuffer(byteBytes);
        ByteBuffer rbuff = MPI.newByteBuffer(byteBytes * numProcs);

        for (int i = 0; i < byteBytes; ++i){
            sbuff.put(i,(byte)'a');
        }

        for (int i = 0; i < byteBytes*numProcs; ++i){
            rbuff.put(i,(byte)'b');
        }


        //comm.allGather(sbuff,maxMsgSize,MPI.CHAR,rbuff,maxMsgSize,MPI.CHAR);
        comm.allGather(rbuff,maxMsgSize,MPI.CHAR);
        MPI.Finalize();
    }
}
