package org.saliya.ompi.omb.collectives;

import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;
import org.saliya.ompi.omb.ParallelOps;
import org.saliya.ompi.util.MpiOps;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * @author Saliya Ekanayake (esaliya at gmail dot com)
 *         Nigel Pugh (nigel dot pugh32 at gmail dot com)
 */

public class OsuBroadcast {
    public static void main(String[] args) throws MPIException, IOException, InterruptedException {
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

        String mmapDir = args[2];
        ParallelOps.setupParallelism(args, maxMsgSize, mmapDir);

        Boolean isMmap = Boolean.parseBoolean(args[3]);

        int byteBytes = maxMsgSize;
        ByteBuffer sbuff = MPI.newByteBuffer(byteBytes);

        String msg = "Rank " + ParallelOps.worldProcRank + " is on " + MPI.getProcessorName() + "\n";
        msg = MpiOps.allReduceStr(msg, ParallelOps.worldProcsComm);
        if (ParallelOps.worldProcRank == 0){
            System.out.println(msg);
            System.out.println("#Bytes\tAvgLatency(us)\tMinLatency(us)\tMaxLatency(us)\t#Itr");
        }

        // TODO - debugs
        boolean stop = false;
        double [] vbuff = new double[1];
        for (int numBytes = 0; numBytes <= maxMsgSize; numBytes = (numBytes == 0 ? 1 : numBytes*2)){
            for (int i = 0; i < byteBytes; ++i){
                /*sbuff.put(i,((byte)'a'));*/
                // TODO - debugs
                if (ParallelOps.worldProcRank == 2){
                    sbuff.put(i,((byte)'b'));
                } else {
                    sbuff.put(i,((byte)'z'));
                }
            }

            if (numBytes > largeMsgSize){
                skip = skipLarge;
                iterations = iterationsLarge;
            }
            ParallelOps.worldProcsComm.barrier();

            double timer = 0.0;
            double tStart, tStop;
            double minLatency, maxLatency, avgLatency;
            for (int i = 0; i < iterations + skip; ++i){
                // TODO - debugs
                /*ParallelOps.worldProcsComm.barrier();*/

                tStart = MPI.wtime();
                if (!isMmap) {
                    ParallelOps.worldProcsComm.bcast(sbuff, numBytes, MPI.BYTE, 2);
                } else {
                    ParallelOps.broadcast(sbuff, numBytes, 2);
                }
                tStop = MPI.wtime();
                if (i >= skip){
                    timer += tStop - tStart;
                }

                // TODO - debugs
                if (numBytes == maxMsgSize) {
                    if (ParallelOps.worldProcRank == 33) {
                        boolean error = false;
                        StringBuilder sb = new StringBuilder();
                        for (int j = 0; j < numBytes; ++j) {
                            char c = (char) sbuff.get(i);
                            if (c != 'b'){
                                System.out.println("Error in allgather ");
                                error = true;
                                break;
                            }
                            /*sb.append((char)sbuff.get(i)).append(' ');*/
                        }
                        /*System.out.println(sb.toString());*/
                        if (!error) {
                            System.out.println("All good");
                        }
                    }
                    stop = true;
                    break;
                }

                ParallelOps.worldProcsComm.barrier();
                // TODO - debugs
                ParallelOps.broadCastCleanup();
            }
            // TODO - debugs
            if (stop) break;

            double latency = (timer *1e6)/iterations;
            vbuff[0] = latency;
            ParallelOps.worldProcsComm.reduce(vbuff,1,MPI.DOUBLE,MPI.MIN,0);
            minLatency = vbuff[0];
            vbuff[0] = latency;
            ParallelOps.worldProcsComm.reduce(vbuff,1,MPI.DOUBLE,MPI.MAX,0);
            maxLatency = vbuff[0];
            vbuff[0] = latency;
            ParallelOps.worldProcsComm.reduce(vbuff,1,MPI.DOUBLE,MPI.SUM,0);
            avgLatency = vbuff[0] / ParallelOps.worldProcsCount;
            if (ParallelOps.worldProcRank == 0){
                System.out.println(numBytes + "\t" + avgLatency +"\t" + minLatency + "\t" + maxLatency + "\t" + iterations);
            }
            ParallelOps.worldProcsComm.barrier();
        }
        ParallelOps.endParallelism();

    }
}
