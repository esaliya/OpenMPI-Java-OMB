package org.saliya.omb.collectives;

import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;
import org.saliya.util.MpiOps;

import java.nio.ByteBuffer;

/**
 * Created by nigelp32 on 7/7/14.
 */
public class OsuBroadcast {
    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        Intracomm comm = MPI.COMM_WORLD;
        int rank = comm.getRank();
        int numProcs = comm.getSize();

        int maxMsgSize = 1048576; // 1MB, i.e. 1024x1024 bytes
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

        int floatBytes = (maxMsgSize/4) * 4;
        ByteBuffer sbuff = MPI.newByteBuffer(floatBytes);
        ByteBuffer rbuff = MPI.newByteBuffer(floatBytes);

        for (int i = 0; i < floatBytes; ++i){
            sbuff.put((byte)1);
            rbuff.put((byte)0);
        }

        String msg = "Rank " + rank + " is on " + MPI.getProcessorName() + "\n";
        msg = MpiOps.allReduceStr(msg, comm);
        if (rank == 0){
            System.out.println(msg);
            System.out.println("#Bytes\tAvgLatency(us)\tMinLatency(us)\tMaxLatency(us)\t#Itr");
        }

        double [] vbuff = new double[1];
        for (int numFloats = 1; numFloats*4 <= maxMsgSize; numFloats *= 2){
            if (numFloats > largeMsgSize){
                skip = skipLarge;
                iterations = iterationsLarge;
            }
            comm.barrier();

            double timer = 0.0;
            double tStart, tStop;
            double minLatency, maxLatency, avgLatency;
            for (int i = 0; i < iterations + skip; ++i){
                tStart = MPI.wtime();
                comm.bcast(sbuff,numFloats,MPI.FLOAT, 0);
                //Use command P to show available parameters//
                tStop = MPI.wtime();
                if (i >= skip){
                    timer += tStop - tStart;
                }
                comm.barrier();
            }
            double latency = (timer *1e6)/iterations;
            vbuff[0] = latency;
            comm.reduce(vbuff,1,MPI.DOUBLE,MPI.MIN,0);
            minLatency = vbuff[0];
            vbuff[0] = latency;
            comm.reduce(vbuff,1,MPI.DOUBLE,MPI.MAX,0);
            maxLatency = vbuff[0];
            vbuff[0] = latency;
            comm.reduce(vbuff,1,MPI.DOUBLE,MPI.SUM,0);
            avgLatency = vbuff[0] / numProcs;
            if (rank == 0){
                System.out.println(numFloats*4 + "\t" + avgLatency +"\t" + minLatency + "\t" + maxLatency + "\t" + iterations);
            }
            comm.barrier();
        }
        MPI.Finalize();

    }
}
