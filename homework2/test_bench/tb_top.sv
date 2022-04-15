`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/08/2022 12:19:15 AM
// Design Name: 
// Module Name: tb_top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module tb_top(

    );

    parameter clock_period = 20;
    reg clk;
    reg reset;
    top top_inst(
        .clk(clk),
        .reset(reset)
        );


    initial begin
        $readmemb("/home/guanrenyang/AI3615-AI-Chip-Design/homework2/lab2/lab2.srcs/sources_1/new/instructions.dat", top_inst.instruction_buffer_du.mem);
        $readmemh("/home/guanrenyang/AI3615-AI-Chip-Design/homework2/lab2/lab2.srcs/sources_1/new/shared_memory_contents.dat", top_inst.shared_memory_du.mem);
        clk=0;
        reset=0;
        
        forever begin
            #clock_period clk=~clk;
        end   
    end

    initial begin
        #clock_period;
        reset=1;   
        #clock_period;
   
    end
  
endmodule
