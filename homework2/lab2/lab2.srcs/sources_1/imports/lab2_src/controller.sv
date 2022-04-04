module  controller  (
    input           clk ,
    input           reset ,
    input start ,
    output [1:0]         SHM_en ,      // shared_memory ren & wen
    output [1:0]         INBUF_en ,    // input buffer ren & wen
    output [1:0]         WBUF_en ,     // weight buffer ren & wen
    output [1:0]         PEARRAY_en ,     // PE pen & cen
    output               ELEARRAY_en,    // elementwise array en
    output               INSBUF_en,    // instruction buffer ren
    output               DECODER_en,  // decoder en

    input [3:0]    func,
    input [9:0]    rs1,
    input [9:0]    rs2,
    input [3:0]    opcode,
    input          compute_finished,

    output [7:0] pc,
    output [3:0 ] cs,
    output [5:0] addr1,
    output [5:0] addr2,
    output [5:0] addr_result,

    output [1:0] source,
    output [1:0] target
    );

    // reg [3:0] func_reg;     // reg to store func code for elementwise array

    //machine state decode
    parameter            READ_INSTR   = 4'd0 ;
    parameter            INSTR_DOCODE  = 4'd1 ;
    parameter            INSTR_DOCODE_POST  = 4'd2 ;
    parameter            PRE_LOAD  = 4'd3 ;
    parameter            PE_COMP =4'd4;
    parameter            ELECOMP_W2SHM = 4'd5 ;
    parameter            MOVE=4'd6;
    parameter            IDLE = 4'd7 ;
    
    //machine variable
    reg [3:0]            st_next ;
    reg [3:0]            st_cur ;
    reg [7:0]            pc_reg;
    reg [2:0]            move_cnt;
    reg [2:0]            preload_cnt;
    reg [2:0]            write_cnt;
    reg [5:0]            result_addr_reg;
    reg [5:0]            addr1_reg;
    reg [5:0]            addr2_reg;
    reg [1:0]            source_reg;
    reg [1:0]            target_reg;

    assign source=source_reg;
    assign target_reg=target;
    assign pc=pc_reg;
    assign addr1=addr1_reg;
    assign addr2=addr2_reg;
    assign result_addr= result_addr_reg;
    assign cs =st_cur;
    
    always@(posedge clk or negedge reset) begin
        if(!reset) begin 
            source_reg<=0;
        end
        else begin
            if(st_cur==INSTR_DOCODE_POST)begin
                if(rs1[9:6]==4'b0001) source_reg<=1;
                else if(rs1[9:6]==4'b0010) source_reg<=2;
                else if(rs1[9:6]==4'b0100) source_reg<=3;
            end

        end
    end
        always@(posedge clk or negedge reset) begin
        if(!reset) begin 
            target_reg<=0;
        end
        else begin
            if(st_cur==INSTR_DOCODE_POST)begin
                if(rs2[9:6]==4'b0001) target_reg<=1;
                else if(rs2[9:6]==4'b0010) target_reg<=2;
                else if(rs2[9:6]==4'b0100)  target_reg<=3;
            end

        end
    end
    
    always@(posedge clk or negedge reset)begin
        if(!reset) pc_reg<=0;
        else begin
            if({func,opcode} !=8'b1111_1111 && st_cur==INSTR_DOCODE_POST)
                pc_reg<=pc_reg+1;
            else 
                pc_reg<=pc_reg;
        end
    end
    always @(posedge clk or negedge reset) begin
        if(!reset) preload_cnt<=0;
        else begin
            if(st_cur!=PRE_LOAD) preload_cnt<=0;
            else begin
                preload_cnt<=preload_cnt+1;
            end
        end
    end
    always @(posedge clk or negedge reset) begin
        if(!reset) move_cnt<=0;
        else begin
            if(st_cur!=MOVE) move_cnt<=0;
            else begin
                move_cnt<=move_cnt+1;
            end
        end
    end
    always @(posedge clk or negedge reset) begin
        if(!reset) write_cnt<=0;
        else begin
            if(st_cur!=ELECOMP_W2SHM) write_cnt<=0;
            else begin
                write_cnt<=write_cnt+1;
            end
        end
    end
    always @(posedge clk or negedge reset) begin
        if(!reset) result_addr_reg<=0;
        else begin
            if(st_cur==PRE_LOAD) result_addr_reg<=rs2[5:0];
        end 
    end
    always @(posedge clk or negedge reset) begin
        if(!reset) result_addr_reg<=0;
        else begin
            if(st_cur==PRE_LOAD) result_addr_reg<=rs2[5:0];
            else if(st_cur==PE_COMP && preload_cnt >=1) 
                result_addr_reg<=result_addr_reg+1;
        end 
    end
    always @(posedge clk or negedge reset) begin
        if(!reset) addr1_reg<=0;
        else begin
            if(st_cur==INSTR_DOCODE_POST) addr1_reg<=rs1[5:0];
            else if(st_cur==MOVE && move_cnt<4) addr1_reg<=addr1_reg+1;
            else if(st_cur==PRE_LOAD && preload_cnt<4) addr1_reg<=addr1_reg+1;
            
        end 
    end
    always @(posedge clk or negedge reset) begin
        if(!reset) addr2_reg<=0;
        else begin
            if(st_cur==INSTR_DOCODE_POST) addr2_reg<=rs2[5:0];
            else if(st_cur==MOVE && move_cnt>=1) addr1_reg<=addr1_reg+1;
            else if(st_cur==PRE_LOAD && preload_cnt>=1) addr2_reg<=addr2_reg+1;
        end 
    end


    //(1) state transfer
    always @(posedge clk or negedge reset) begin
        if (!reset) begin
            st_cur      <= 'b0 ;
        end
        else begin
            st_cur      <= st_next ;
        end
    end
    //(2) state switch, using block assignment for combination-logic   
    always @(*) begin
        st_next = st_cur ;
        case(st_cur)
            IDLE:  begin
                if(start) st_next= READ_INSTR;
            end
            READ_INSTR:begin
                st_next= INSTR_DOCODE;
            end
            INSTR_DOCODE:begin
                st_next=INSTR_DOCODE_POST;
            end
            INSTR_DOCODE_POST: begin 
                case({func,opcode})
                    8'b00010010:st_next=PRE_LOAD;
                    8'b00100010:st_next=PRE_LOAD;
                    8'b00010001:st_next=MOVE;
                    8'b00010100:st_next=PE_COMP;
                    8'b11111111:st_next=IDLE;
                    default:st_next=IDLE;
                endcase
            end
            PRE_LOAD:begin
                if (preload_cnt==4) st_next= READ_INSTR;
            end
            MOVE:begin
                if(move_cnt==4) st_next=READ_INSTR;
            end
            PE_COMP:begin
                if(compute_finished)
                st_next=ELECOMP_W2SHM;
            end
            
            ELECOMP_W2SHM:begin
                if(write_cnt==4) st_next=READ_INSTR;
            end
    

            default:   begin st_next = IDLE ;
            end
        endcase
    end

    //(3) output logic


    reg [1:0]         reg_SHM_en ;     // shared_memory ren & wen
    always @(*) begin
        reg_SHM_en=2'b0;
        if(st_cur==PRE_LOAD && preload_cnt <4 )begin
            reg_SHM_en=2'b10;
        end
        else if (st_cur==MOVE && move_cnt <4)begin
            reg_SHM_en=2'b10;
        end
        else if(st_cur==ELECOMP_W2SHM && write_cnt>=1)begin
             reg_SHM_en=2'b01;
        end
        else begin
            reg_SHM_en=2'b00;
        end
    end







    reg [1:0]         reg_INBUF_en ;   // input buffer ren & wen
    always @ (*) begin
         reg_INBUF_en=2'b0;
        if(st_cur==MOVE && move_cnt>=1) begin
            if(rs2[9:6]==4'b0010)  begin
              reg_INBUF_en=2'b01;
            end
        end
        else if(st_cur==PE_COMP)begin
            reg_INBUF_en=2'b10;
        end
    end






    reg [1:0]         reg_WBUF_en ;     // weight buffer ren & wen
    always @(*) begin
         reg_WBUF_en=2'b0;
        if(st_cur==MOVE && move_cnt>=1) begin
            if(rs2[9:6]==4'b0100)  begin
              reg_WBUF_en=2'b01;
            end
        end
        else if(st_cur==PE_COMP)begin
            reg_WBUF_en=2'b10;
        end
    end




    reg [1:0]         reg_PEARRAY_en ;     // PE pen & cen
    always @(*) begin
        reg_PEARRAY_en=2'b0;
        if(st_cur==PRE_LOAD && preload_cnt>=1) begin
            reg_PEARRAY_en=2'b10;
        end 
        else if(st_cur==PE_COMP) begin
            reg_PEARRAY_en=2'b01;
        end
    end



    reg               reg_ELEARRAY_en;    // elementwise array en
    always @(*) begin
        if(st_cur==ELECOMP_W2SHM && write_cnt<4)
            reg_ELEARRAY_en=1'b1;
        else 
            reg_ELEARRAY_en=1'b0;
    end




    reg               reg_INSBUF_en;    // instruction buffer ren
    always @(*) begin

            if(st_cur==READ_INSTR)  
                reg_INSBUF_en=1'b1;
            else 
                reg_INSBUF_en=1'b0;

    end


    reg               reg_DECODER_en;    
    always @(*) begin

            if(st_cur==INSTR_DOCODE)  
                reg_INSBUF_en=1'b1;
            else 
                reg_DECODER_en=1'b0;
    end




    assign    SHM_en      =  reg_SHM_en ;
    assign    INBUF_en    =  reg_INBUF_en ;
    assign    WBUF_en     =  reg_WBUF_en ;
    assign    PEARRAY_en  =  reg_PEARRAY_en ;
    assign    ELEARRAY_en =  reg_ELEARRAY_en ;
    assign    INSBUF_en   =  reg_INSBUF_en ;
    assign    DECODER_en   =  reg_DECODER_en ;

endmodule