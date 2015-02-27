/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hit.ui;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author songmingye
 */
public class Core {
    public static void main(String[] ars){
        Core p = new Core();
        p.get_results();
        for(int i=0;i<10;i++){
            System.out.println(p.results[i]);
        }
    } 
    public int now_num;
    public int[] results;
    public Core(){
        now_num=0;
        results = new int[10000];
        get_results();
    }
    public void next(){
        now_num++;
        if(now_num>9999){
            now_num = 0;
        }
    }
    public void before(){
        now_num--;
        if(now_num<0){
            now_num = 9999;
        }
    }
    private void get_results(){
        try {
            FileReader fr=new FileReader("./Result labels.txt");
            BufferedReader br=new BufferedReader(fr);
            String s;
            int no=0;
            while((s = br.readLine())!=null){
                int co = Integer.parseInt(s);
                results[no] = co;
                no++;
            }
        } catch (IOException ex) {
            Logger.getLogger(Core.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
