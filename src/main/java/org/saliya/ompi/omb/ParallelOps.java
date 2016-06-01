package org.saliya.ompi.omb;

import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;
import net.openhft.lang.io.ByteBufferBytes;
import net.openhft.lang.io.Bytes;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.HashSet;
import java.util.regex.Pattern;

public class ParallelOps {
    public static String machineName;
    public static int nodeCount=1;
    public static int threadCount=1;

    public static int nodeId;

    public static Intracomm worldProcsComm;
    public static int worldProcRank;
    public static int worldProcsCount;
    public static int worldProcsPerNode;
    public static int worldProcRankLocalToNode;

    public static Intracomm mmapProcComm;
    // Number of memory mapped groups per process
    public static int mmapsPerNode;
    public static String mmapScratchDir;
    public static int mmapIdLocalToNode;
    public static int mmapProcRank;
    public static int mmapProcsCount;
    public static boolean isMmapLead;
    public static boolean isMmapHead = false;
    public static boolean isMmapTail = false;
    public static int[] mmapProcsWorldRanks;
    public static int mmapLeadWorldRank;
    public static int mmapLeadWorldRankLocalToNode;
    public static int mmapProcsRowCount;

    // mmap leaders form one communicating group and the others (followers)
    // belong to another communicating group.
    public static Intracomm cgProcComm;
    public static int cgProcRank;
    public static int cgProcsCount;

    public static String mmapCollectiveFileName;
    public static String mmapCollectiveFileName2;
    public static String mmapLockFileNameOne;
    public static String mmapLockFileNameTwo;
    public static Bytes mmapLockOne;
    public static Bytes mmapLockTwo;
    public static Bytes mmapCollectiveBytes;
    public static Bytes mmapCollectiveBytes2;
    public static ByteBuffer mmapCollectiveByteBuffer;
    public static ByteBuffer mmapCollectiveByteBuffer2;

    public static Bytes mmapWriteBytes;
    public static Bytes mmapReadBytes;
    public static ByteBuffer mmapWriteByteBuffer;
    public static ByteBuffer mmapReadByteBuffer;

    private static IntBuffer intBuffer;

    private static int LOCK = 0;
    private static int FLAG = Long.BYTES;
    private static int COUNT = 2*Long.BYTES;

    private static HashMap<Integer, Integer> cgProcCommRankOfMmapLeaderForRank;

    public static void setupParallelism(String[] args, int maxMsgSize, String mmapDir) throws MPIException, IOException {
        MPI.Init(args);
        machineName = MPI.getProcessorName();
        mmapsPerNode = 1;
        mmapScratchDir = mmapDir;

        intBuffer = MPI.newIntBuffer(1);

        worldProcsComm = MPI.COMM_WORLD; //initializing MPI world communicator
        worldProcRank = worldProcsComm.getRank();
        worldProcsCount = worldProcsComm.getSize();

        /* Create communicating groups */
        worldProcsPerNode = worldProcsCount / nodeCount;
        boolean heterogeneous = (worldProcsPerNode * nodeCount) != worldProcsCount;

        /* Logic to identify how many processes are within a node and
        *  the q and r values. These are used to processes to mmap groups
        *  within a node.
        *
        *  Important: the code assumes continues rank distribution
        *  within a node. */
        int[] qr = findQandR();
        int q = qr[0];
        int r = qr[1];

        // Memory mapped groups and communicating groups
        mmapIdLocalToNode =
                worldProcRankLocalToNode < r * (q + 1)
                        ? worldProcRankLocalToNode / (q + 1)
                        : (worldProcRankLocalToNode - r) / q;
        mmapProcsCount = worldProcRankLocalToNode < r*(q+1) ? q+1 : q;


        // Communicator for processes within a  memory map group
        mmapProcComm = worldProcsComm.split((nodeId*mmapsPerNode)+mmapIdLocalToNode, worldProcRank);
        mmapProcRank = mmapProcComm.getRank();

        isMmapLead = mmapProcRank == 0;
        isMmapHead = isMmapLead; // for chain calls
        isMmapTail = (mmapProcRank == mmapProcsCount - 1); // for chain calls
        mmapProcsWorldRanks = new int[mmapProcsCount];
        mmapLeadWorldRankLocalToNode =
                isMmapLead
                        ? worldProcRankLocalToNode
                        : (q * mmapIdLocalToNode + (mmapIdLocalToNode < r
                        ? mmapIdLocalToNode
                        : r));
        mmapLeadWorldRank = worldProcRank - (worldProcRankLocalToNode
                - mmapLeadWorldRankLocalToNode);
        // Assumes continues ranks within a node
        for (int i = 0; i < mmapProcsCount; ++i){
            mmapProcsWorldRanks[i] = mmapLeadWorldRank +i;
        }

        // Leaders talk, their color is 0
        // Followers will get a communicator of color 1,
        // but will make sure they don't talk ha ha :)
        cgProcComm = worldProcsComm.split(isMmapLead ? 0 : 1, worldProcRank);
        cgProcRank = cgProcComm.getRank();
        cgProcsCount = cgProcComm.getSize();

        boolean status = new File(mmapScratchDir).mkdirs();
        /* Allocate memory maps for collective communications like AllReduce and Broadcast */
        mmapCollectiveFileName = machineName + ".mmapId." + mmapIdLocalToNode + ".mmapCollective.bin";
        mmapCollectiveFileName2 = machineName + ".mmapId." + mmapIdLocalToNode + ".mmapCollective2.bin";
        mmapLockFileNameOne = machineName + ".mmapId." + mmapIdLocalToNode + ".mmapLockOne.bin";
        mmapLockFileNameTwo = machineName + ".mmapId." + mmapIdLocalToNode + ".mmapLockTwo.bin";
        try (FileChannel mmapCollectiveFc = FileChannel
                .open(Paths.get(mmapScratchDir, mmapCollectiveFileName),
                        StandardOpenOption.CREATE, StandardOpenOption.READ,
                        StandardOpenOption.WRITE);
             FileChannel mmapCollectiveFc2 = FileChannel
                     .open(Paths.get(mmapScratchDir, mmapCollectiveFileName2),
                             StandardOpenOption.CREATE, StandardOpenOption.READ,
                             StandardOpenOption.WRITE)) {

            mmapCollectiveBytes = ByteBufferBytes.wrap(mmapCollectiveFc.map(
                    FileChannel.MapMode.READ_WRITE, 0L, maxMsgSize*mmapProcsCount));
            mmapCollectiveBytes2 = ByteBufferBytes.wrap(mmapCollectiveFc2.map(
                    FileChannel.MapMode.READ_WRITE, 0L, ((long)maxMsgSize)*worldProcsCount));
            mmapCollectiveByteBuffer = mmapCollectiveBytes.sliceAsByteBuffer(mmapCollectiveByteBuffer);
            mmapCollectiveByteBuffer2 = mmapCollectiveBytes2.sliceAsByteBuffer(mmapCollectiveByteBuffer2);

            if (isMmapLead){
                for (int i = 0; i < maxMsgSize; ++i) {
                    mmapCollectiveBytes.writeByte(i, 0);
                    mmapCollectiveBytes2.writeByte(i, 0);
                }
            }

            File lockFile = new File(mmapScratchDir, mmapLockFileNameOne);
            FileChannel fc = new RandomAccessFile(lockFile, "rw").getChannel();
            mmapLockOne = ByteBufferBytes.wrap(fc.map(FileChannel.MapMode.READ_WRITE, 0, 64));
            if (isMmapLead){
                mmapLockOne.writeBoolean(FLAG, false);
                mmapLockOne.writeLong(COUNT, 0);
            }

            lockFile = new File(mmapScratchDir, mmapLockFileNameTwo);
            fc = new RandomAccessFile(lockFile, "rw").getChannel();
            mmapLockTwo = ByteBufferBytes.wrap(fc.map(FileChannel.MapMode.READ_WRITE, 0, 64));
            if (isMmapLead){
                mmapLockTwo.writeBoolean(FLAG, false);
                mmapLockTwo.writeLong(COUNT, 0);
            }
        }

        cgProcCommRankOfMmapLeaderForRank = new HashMap<>(worldProcsCount);
        String mmapWriteFileName = machineName + ".mmapId." + mmapIdLocalToNode + ".mmapWrite.bin";
        String mmapReadFileName = machineName + ".mmapId." + mmapIdLocalToNode + ".mmapRead.bin";
        try (FileChannel mmapWriteFc = FileChannel
                .open(Paths.get(mmapScratchDir, mmapWriteFileName),
                        StandardOpenOption.CREATE, StandardOpenOption.READ,
                        StandardOpenOption.WRITE);
             FileChannel mmapReadFc = FileChannel
                     .open(Paths.get(mmapScratchDir, mmapReadFileName),
                             StandardOpenOption.CREATE, StandardOpenOption.READ,
                             StandardOpenOption.WRITE)) {

            mmapWriteBytes = ByteBufferBytes.wrap(mmapWriteFc.map(FileChannel.MapMode.READ_WRITE, 0L, 3 * Integer.BYTES));
            mmapReadBytes = ByteBufferBytes.wrap(
                    mmapReadFc.map(FileChannel.MapMode.READ_WRITE, 0L, 3 * worldProcsCount * Integer.BYTES));
            mmapWriteByteBuffer = mmapWriteBytes.sliceAsByteBuffer(mmapWriteByteBuffer);
            mmapReadByteBuffer = mmapReadBytes.sliceAsByteBuffer(mmapReadByteBuffer);
        }

        findCgProcCommRankOfMmapLeadForAllRanks();
    }

    public static void endParallelism() throws MPIException {
        MPI.Finalize();
    }

    private static void findCgProcCommRankOfMmapLeadForAllRanks() throws MPIException {
        if (isMmapLead){
            mmapWriteBytes.writeInt(0, cgProcRank);
            mmapWriteBytes.writeInt(Integer.BYTES, worldProcRank);
        }
        if (isMmapTail){
            mmapWriteBytes.writeInt(2*Integer.BYTES, worldProcRank);
        }
        worldProcsComm.barrier();
        if(isMmapLead){
            cgProcComm.allGather(mmapWriteByteBuffer, 3, MPI.INT, mmapReadByteBuffer, 3, MPI.INT);
        }
        worldProcsComm.barrier();
        int cgr;
        int fromWorldRank, toWorldRank;
        int offset;
        for (int i = 0; i < worldProcsCount; ++i){
            offset = 3*i*Integer.BYTES;
            cgr = mmapReadBytes.readInt(offset);
            fromWorldRank = mmapReadBytes.readInt(offset+Integer.BYTES);
            toWorldRank = mmapReadBytes.readInt(offset+2*Integer.BYTES);
            for (int j = fromWorldRank; j <=toWorldRank; ++j){
                cgProcCommRankOfMmapLeaderForRank.put(j, cgr);
            }
        }

    }

    private static int[] findQandR() throws MPIException {
        int q,r;
        String str = worldProcRank+ "@" +machineName +'#';
        intBuffer.put(0, str.length());
        worldProcsComm.allReduce(intBuffer, 1, MPI.INT, MPI.MAX);
        int maxLength = intBuffer.get(0);
        CharBuffer buffer = MPI.newCharBuffer(maxLength*worldProcsCount);
        buffer.position(maxLength*worldProcRank);
        buffer.put(str);
        for (int i = str.length(); i < maxLength; ++i){
            buffer.put(i, '~');
        }

        worldProcsComm.allGather(buffer, maxLength, MPI.CHAR);
        buffer.position(0);
        Pattern nodeSep = Pattern.compile("#~*");
        Pattern nameSep = Pattern.compile("@");
        String[] nodeSplits = nodeSep.split(buffer.toString());
        HashMap<String, Integer> nodeToProcCount = new HashMap<>();
        HashMap<Integer, String> rankToNode = new HashMap<>();
        String node;
        int rank;
        String[] splits;
        for(String s: nodeSplits){
            splits = nameSep.split(s);
            rank = Integer.parseInt(splits[0].trim());
            node = splits[1].trim();
            if (nodeToProcCount.containsKey(node)){
                nodeToProcCount.put(node, nodeToProcCount.get(node)+1);
            } else {
                nodeToProcCount.put(node, 1);
            }
            rankToNode.put(rank, node);
        }

        // The following logic assumes MPI ranks are continuous within a node
        String myNode = rankToNode.get(worldProcRank);
        HashSet<String> visited = new HashSet<>();
        int rankOffset=0;
        nodeId = 0;
        for (int i = 0; i < worldProcRank; ++i){
            node = rankToNode.get(i);
            if (visited.contains(node)) continue;
            visited.add(node);
            if (node.equals(myNode)) break;
            ++nodeId;
            rankOffset += nodeToProcCount.get(node);
        }
        worldProcRankLocalToNode = worldProcRank - rankOffset;
        final int procCountOnMyNode = nodeToProcCount.get(myNode);
        q = procCountOnMyNode / mmapsPerNode;
        r = procCountOnMyNode % mmapsPerNode;

        return new int[]{q,r};
    }

    public static void broadcast(ByteBuffer buffer, int length, int root) throws MPIException, InterruptedException {


        /* for now let's assume a second invocation of broadcast will NOT happen while some ranks are still
        *  doing the first invocation. If that happens, current implementation can screw up */

        Object obj = cgProcCommRankOfMmapLeaderForRank.get(root);
        // TODO - debugs
        System.out.println("Rank: " + worldProcRank + " came to bcast " + " is obj null " + (obj == null));
        int cgProcRankOfMmapLeaderForRoot =  (int)obj;
        if (root == worldProcRank){
            /* I am the root and I've the content, so write to my shared buffer */
            mmapCollectiveBytes.position(0);
            buffer.position(0);
            for (int i = 0; i < length; ++i) {
                mmapCollectiveBytes.writeByte(i, buffer.get(i));
            }
//            mmapLockOne.busyLockLong(LOCK);
            mmapLockOne.writeInt(COUNT,1); // order matters as we don't have locks now
            mmapLockOne.writeBoolean(FLAG, true);
            /*mmapLockOne.writeInt(COUNT, 1);*/
//            mmapLockOne.unlockLong(LOCK);

            if (!isMmapLead) return;
        }

        if (root != worldProcRank && isRankWithinMmap(root) && !isMmapLead){
            // TODO - debugs
            System.out.println("Rank: " + worldProcRank + " came into second if bcast ");
            /* I happen to be within the same mmap as root and I am not an mmaplead,
            so read from shared buffer if root is done writing to it */
            boolean ready = false;
            int count;
            while (!ready){
//                mmapLockOne.busyLockLong(LOCK);
                ready = mmapLockOne.readBoolean(FLAG);
                if (ready) {
                    /*count = mmapLockOne.readInt(COUNT);
                    ++count;
                    mmapLockOne.writeInt(COUNT, count);*/
                    count = mmapLockOne.addAndGetInt(COUNT,1);
                    if (count == mmapProcsCount && worldProcsCount == mmapProcsCount){
                        mmapLockOne.writeBoolean(FLAG, false);
                        mmapLockOne.writeInt(COUNT, 0);
                    }
                }
//                mmapLockOne.unlockLong(LOCK);
            }
        } else {
            // TODO - debugs
            System.out.println("Rank: " + worldProcRank + " came into second else bcast ");
            if (ParallelOps.isMmapLead) {
                if (root == worldProcRank) {
                    boolean ready = false;
                    int count;
                    while (!ready) {
//                        mmapLockOne.busyLockLong(LOCK);
                        ready = mmapLockOne.readBoolean(FLAG);
                        if (ready) {
                            /*count = mmapLockOne.readInt(COUNT);
                            ++count;
                            mmapLockOne.writeInt(COUNT, count);*/
                            count = mmapLockOne.addAndGetInt(COUNT, 1);
                            if (count == mmapProcsCount) {
                                mmapLockOne.writeBoolean(FLAG, false);
                                mmapLockOne.writeInt(COUNT, 0);
                            }
                        }
//                        mmapLockOne.unlockLong(LOCK);
                    }
                }
                cgProcComm.bcast(mmapCollectiveByteBuffer, length, MPI.BYTE, cgProcRankOfMmapLeaderForRoot);
                if (root != worldProcRank) {
//                    mmapLockTwo.busyLockLong(LOCK);
                    mmapLockTwo.writeInt(COUNT, 1); // order matters as we don't have locks now
                    mmapLockTwo.writeBoolean(FLAG, true);
//                    mmapLockTwo.unlockLong(LOCK);
                }
            } else {
                boolean ready = false;
                int count;
                while (!ready) {
//                    mmapLockTwo.busyLockLong(LOCK);
                    ready = mmapLockTwo.readBoolean(FLAG);
                    if (ready) {
                        /*count = mmapLockTwo.readInt(COUNT);
                        ++count;
                        mmapLockTwo.writeInt(COUNT, count);*/
                        count = mmapLockTwo.addAndGetInt(COUNT, 1);
                        if (count == mmapProcsCount) {
                            mmapLockTwo.writeBoolean(FLAG, false);
                            mmapLockTwo.writeInt(COUNT, 0);
                        }
                    }
//                    mmapLockTwo.unlockLong(LOCK);
                }
            }
        }

        if (root != worldProcRank){
            mmapCollectiveBytes.position(0);
            buffer.position(0);
            mmapCollectiveBytes.read(buffer, length);
            /*for (int i = 0; i < length; ++i){
                buffer.put(i,mmapCollectiveBytes.readByte(i));
            }*/
        }
    }

    public static void broadCastCleanup() throws MPIException, InterruptedException {
        worldProcsComm.barrier();
        if (isMmapLead){
            mmapLockOne.busyLockLong(LOCK);
            mmapLockOne.writeBoolean(FLAG, false);
            mmapLockOne.writeInt(COUNT, 0);
            mmapLockOne.unlockLong(LOCK);

            mmapLockTwo.busyLockLong(LOCK);
            mmapLockTwo.writeBoolean(FLAG, false);
            mmapLockTwo.writeInt(COUNT, 0);
            mmapLockTwo.unlockLong(LOCK);
        }
        worldProcsComm.barrier();
    }

    private static boolean isRankWithinMmap(int rank){
        return (mmapLeadWorldRank <= rank && rank <= (mmapLeadWorldRank+mmapProcsCount));
    }

    public static void allGather(ByteBuffer sbuff, int numBytes, ByteBuffer rbuff) throws MPIException {
        int offset = mmapProcRank*numBytes;
        for (int i = 0; i < numBytes; ++i){
            mmapCollectiveBytes.writeByte(offset+i, sbuff.get(i));
        }
        worldProcsComm.barrier();
        if(isMmapLead){
            cgProcComm.allGather(mmapCollectiveByteBuffer, numBytes*mmapProcsCount, MPI.BYTE, mmapCollectiveByteBuffer2, numBytes*mmapProcsCount, MPI.BYTE);
        }
        worldProcsComm.barrier();
        for (int i = 0; i < numBytes*mmapProcsCount; ++i){
            rbuff.put(i, mmapCollectiveBytes2.readByte(i));
        }
    }
}
