package LabelEncoding;

import java.awt.Toolkit;
import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class RandDel {
	
	public static void randomDelete( ) throws IOException{
		String s1="/Users/dutianming/theano/bitbucket_deeplearning/make-ipinyou-data/2997/train.fm.txt";//"/Users/dutianming/Desktop/rtb/data/data_train.txt";
		String s2="/Users/dutianming/theano/bitbucket_deeplearning/make-ipinyou-data/2997/train.fm.txt.mod4.txt";//"/Users/dutianming/Desktop/rtb/data/data_train_small.txt"
		LineNumberReader br = null;
		PrintWriter bw = null;
		br = new LineNumberReader(new FileReader(new File(s1)));
	    bw = new PrintWriter(new FileWriter(new File(s2), true));
	    String line = br.readLine();
	    int row=1;
	    int len=0;
	    while (line != null ) { 
	    	String s=line.split(",")[0].toString();
	    	if(s.equals("1")){
	    		//System.out.println(line);
	    		bw.write(line);
	    		bw.write("\n");
	    	}
	    	else if(row %10==0){
	    		bw.write(line);
	    		bw.write("\n");
	    	}
	        line = br.readLine();
	        row=row+1;
	    }
	    Toolkit.getDefaultToolkit().beep();
	    System.out.println("random delete end");
	    //System.out.println(br.getLineNumber());
	    br.close();
	    br = null;
	    bw.flush();            
	    bw.close();
	    bw = null;
	}
	
	public static void main(String[] args) throws IOException {
		randomDelete();
	}

}
