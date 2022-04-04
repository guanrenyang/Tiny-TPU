module top
(
    input  wire         clk,
    input  wire         reset
    );

    parameter            READ_INSTR   = 4'd0 ;
    parameter            INSTR_DOCODE  = 4'd1 ;
    parameter            INSTR_DOCODE_POST  = 4'd2 ;
    parameter            PRE_LOAD  = 4'd3 ;
    parameter            PE_COMP =4'd4;
    parameter            ELECOMP_W2SHM = 4'd5 ;
    parameter            MOVE=4'd6;
    parameter            IDLE = 4'd7 ;


    // some glue jobs
    wire [1:0]         SHM_en ;      // shared_memory ren & wen
    wire [1:0]         INBUF_en ;    // input buffer ren & wen
    wire [1:0]         WBUF_en ;     // weight buffer ren & wen
    wire [1:0]         PEARRAY_en ;     // PE pen & cen
    wire               ELEARRAY_en;    // elementwise array en
    wire               INSBUF_en;    // instruction buffer ren
    wire               DECODER_en;   // decoder en

    wire [27:0] instruction;
    wire [7:0]pc;
    wire [3:0]    func;
    wire [9:0]    rs1;
    wire [9:0]    rs2;
    wire [3:0]    opcode;
    wire start;
    wire compute_finished;
    
    wire [3:0 ] cs;
    wire [5:0] addr1;
    wire [5:0] addr2;
    wire [5:0] addr_result;



    reg [5:0]  weight_buffer_a;
    reg [127:0] weight_buffer_d;
    wire [127:0] weight_buffer_q;
    reg [5:0]  shared_memory_a;
    reg [127:0] shared_memory_d;
    wire [127:0] shared_memory_q;
    reg [5:0]   input_buffer_a;
    reg [127:0] input_buffer_d;
    wire [127:0] input_buffer_q;
    wire [1:0] source;
    wire [1:0] target;
    reg signed[31:0]PE_array_in_weight[3:0];             // wire from weight buffer direction
    reg signed[31:0]PE_array_in_input[3:0];
    wire signed[31:0]PE_array_result[3:0];

    reg signed[31:0] elementwise_array_in [3:0];
    wire signed [31:0] elementwise_array_out [3:0];
  always @(*) begin
      shared_memory_a=0;
      shared_memory_d=0;
      if(cs==PRE_LOAD && source==1) begin
          shared_memory_a=addr1;
      end
      else if( cs==MOVE && source==1) begin
          shared_memory_a=addr1;
      end
      else if(cs==ELECOMP_W2SHM) begin
          shared_memory_a=addr_result;
          shared_memory_d={elementwise_array_out[3],elementwise_array_out[2],elementwise_array_out[1],elementwise_array_out[0]};
      end
  end
  always @(*) begin
      elementwise_array_in={0,0,0,0};
      if(cs==ELECOMP_W2SHM) 
      elementwise_array_in=PE_array_result;
  end
    always @(*) begin
      input_buffer_a=0;
      input_buffer_d=0;
     if( cs==MOVE && target==1) begin
          input_buffer_a=addr2;
          input_buffer_d=shared_memory_q;
      end
      else if(cs==PE_COMP && source==1) begin
          input_buffer_a=addr1;
          
      end
  end
    always @(*) begin
      weight_buffer_a=0;
      weight_buffer_d=0;
     if( cs==MOVE && target==2) begin
          weight_buffer_a=addr2;
          weight_buffer_d=shared_memory_q;
      end
      else if(cs==PE_COMP && target==2) begin
          weight_buffer_a=addr2;
          
      end
  end
    always @(*) begin
      PE_array_in_input={0,0,0,0};
      PE_array_in_weight={0,0,0,0};
     if(cs==PE_COMP ) begin
          PE_array_in_weight={weight_buffer_q[127:96],weight_buffer_q[95:64],weight_buffer_q[63:32],weight_buffer_q[31:0]};
          PE_array_in_input={input_buffer_q[127:96],input_buffer_q[95:64],input_buffer_q[63:32],input_buffer_q[31:0]};
      end
      else if (cs==PRE_LOAD)
           PE_array_in_weight={shared_memory_q[127:96],shared_memory_q[95:64],shared_memory_q[63:32],shared_memory_q[31:0]};
  end

 
decoder decoder_du
(
    .d(instruction),
    .clk(clk),
    .reset(reset),
    .en(DECODER_en),

    .func(func),
    .rs1(rs1),
    .rs2(rs2),
    .opcode(opcode)
    );

controller controller_du (
    .clk(clk),
    .reset(reset),
    .start(start),    
    .SHM_en(SHM_en) ,      // shared_memory ren & wen
    .INBUF_en(INBUF_en) ,    // input buffer ren & wen
    .WBUF_en(WBUF_en) ,     // weight buffer ren & wen
    .PEARRAY_en(PEARRAY_en) ,     // PE pen & cen
    .ELEARRAY_en(ELEARRAY_en),    // elementwise array en
    .INSBUF_en(INSBUF_en),    // instruction buffer ren
    .DECODER_en(DECODER_en),  // decoder en

    .func(func),
    .rs1(rs1),
    .rs2(rs2),
    .opcode(opcode),
    .pc(pc),
    .compute_finished(compute_finished),

    .cs(cs),
    .addr1(addr1),
    .addr2(addr2),
    .addr_result(addr_result),
    .source(source),
    .target(target)
    );
 elementwise_array  #( 4) elementwise_array_du
(
	// interface to system
    .clk(clk),
    .reset(reset),
    .en(ELEARRAY_en),
    .func(func),                     

    .in (elementwise_array_in),
    .out (elementwise_array_out)

	);

    PE_array #( 4)PE_array_du
(
	// interface to system
    .clk(clk),
    .reset(reset),
    .c_en(PEARRAY_en[0]),                              // compute enable
    .p_en(PEARRAY_en[1]),                              // preload enable
    // interface to PE row .....

    .in_weight(PE_array_in_weight),             // wire from weight buffer direction

    .in_input(PE_array_in_input),              // wire from input buffer direction
    .result(PE_array_result),
    .compute_finished(compute_finished)

	);
    weight_buffer #(8)weight_buffer_du
(
    .q(weight_buffer_q),
    .clk(clk),
    .reset(reset),
    .ren(WBUF_en[1]),
    .wen(WBUF_en[0]),
    .a(weight_buffer_a),
    .d(weight_buffer_d)
    );
shared_memory #(64)shared_memory_du
(
    .q(shared_memory_q),
    .clk(clk),
    .reset(reset),
    .ren(SHM_en[1]),
    .wen(SHM_en[0]),
    .a(shared_memory_a),
    .d(shared_memory_d)
    );
instruction_buffer  #(256)instruction_buffer_du
(
   .q(instruction),
    .clk(clk),
    .reset(reset),
    .ren(INBUF_en[1]),
    .a(pc)
    );
 input_buffer  #( 8)input_buffer_du
(
    .q(input_buffer_q),
    .clk(clk),
    .reset(reset),
    .ren(INBUF_en[1]),
    .wen(INBUF_en[0]),
    .a(input_buffer_a),
    .d(input_buffer_d)
    );
endmodule