package org.saliya.omb.p2p;

import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;
import org.saliya.util.MpiOps;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;

/**
 * @author Saliya Ekanayake (esaliya at gmail dot com)
 */

public class OsuLaten {
    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        Intracomm comm = MPI.COMM_WORLD;
        int rank = comm.getRank();
        int numProcs = comm.getSize();

        if (numProcs != 2){
            System.out.println("This test requires exactly two processes\n");
            MPI.Finalize();
            return;
        }

        int maxMsgSize = 1<<20; // 1MB, i.e. 1024x1024 bytes
        int largeMsgSize = 8192;
        int skip = 200;
        int skipLarge = 10;
        int iterations = 1000;
        int iterationsLarge = 100;

        if (args.length >= 1){
            maxMsgSize = Integer.parseInt(args[0]);
        }

        if (args.length == 2){
            iterations = iterationsLarge = Integer.parseInt(args[1]);
        }

        int byteBytes = maxMsgSize;
        ByteBuffer sbuff = MPI.newByteBuffer(byteBytes);
        ByteBuffer rbuff = MPI.newByteBuffer(byteBytes);

        String msg = "Rank " + rank + " is on " + MPI.getProcessorName() + "\n";
        msg = MpiOps.allReduceStr(msg, comm);
        if (rank == 0){
            System.out.println(msg);
            System.out.println("#Bytes\tLatency(us)");
            System.out.println("Max message size in chars (casted to bytes): " + maxMsgSize + " in bytes:" + maxMsgSize);
        }

        for (int numBytes = 0; numBytes <= maxMsgSize; numBytes = (numBytes == 0 ? 1 : numBytes*2)){
            /* initialize buffers for each run*/
            for (int i = 0; i < byteBytes; ++i){
                sbuff.put(i,(byte)'a');
                rbuff.put(i,(byte)'b');
            }

            if (numBytes > largeMsgSize){
                skip = skipLarge;
                iterations = iterationsLarge;
            }
            comm.barrier();

            double tStart = 0.0, tStop = 0.0;

            if (rank == 0){
                for (int i = 0; i < iterations + skip; ++i){
                    if (i == skip){
                        tStart = MPI.wtime();
                    }
                    comm.send(sbuff,numBytes,MPI.BYTE,1,1);
                    comm.recv(rbuff,numBytes,MPI.BYTE,1,1);
                }
                tStop = MPI.wtime();
            } else if (rank == 1){
                for (int i = 0; i < iterations + skip; ++i){
                    comm.recv(rbuff,numBytes,MPI.BYTE,0,1);
                    comm.send(sbuff,numBytes,MPI.BYTE,0,1);
                }
            }


            if (rank == 0){
                double latency = (tStop - tStart)*1e6 /(2.0 * iterations);
                System.out.println(numBytes + "\t" + new BigDecimal(latency).setScale(2, RoundingMode.UP).doubleValue());
            }
        }
        MPI.Finalize();
    }
}
