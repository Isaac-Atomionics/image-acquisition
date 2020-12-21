from artiq.experiment import *

# pulse = 10*ms  # SET PULSE WIDTH

Initial_freq = 226.5 # Cooling beam at MOT stage
Final_freq = 234 #At the end of Sub-Doppler
freq_step = 1.0 # MHz - Frequency step for cooling beam ramp
time_step = 1.0 # ms - Each step time 
steps = 10 # Number of steps for the ramp 
amp_step = 0.2


class DDSTest(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("urukul0_cpld")
        self.dds0 = self.get_device("urukul0_ch0")
        self.dds1 = self.get_device("urukul0_ch1")
        self.dds2 = self.get_device("urukul0_ch2")
        self.dds3 = self.get_device("urukul0_ch3")
        self.dds4 = self.get_device("urukul1_ch0")
        self.dds5 = self.get_device("urukul1_ch1")

        self.dummy = self.get_device("ttl6")
        self.cameratrigger = self.get_device("ttl4")
        self.coolshutter = self.get_device("ttl8")
        self.probeshutter = self.get_device("ttl9")
        self.eomshutter = self.get_device("ttl10")
        self.motcoiltrigger = self.get_device("ttl7")
        self.shimtrigger = self.get_device("ttl12")
        self.verticalshim = self.get_device("ttl14")
        
        # Shutters: rising edge = open, falling edge = closed
        # MOT coils: falling edge = on, rising edge = off
      

        
    @kernel
    def run(self):
        self.core.reset() 
        
        self.dds0.cpld.init()
        self.dds0.init()
        self.dds0.set_att(0.0)
        
        self.dds1.cpld.init()
        self.dds1.init()
        self.dds1.set_att(0.0)
        
        self.dds2.cpld.init()
        self.dds2.init()
        self.dds2.set_att(0.0)
        
        self.dds3.cpld.init()
        self.dds3.init()
        self.dds3.set_att(0.0)
        
        self.dds4.cpld.init()
        self.dds4.init()
        self.dds4.set_att(0.0)
        
        self.dds5.cpld.init()
        self.dds5.init()
        self.dds5.set_att(0.0)
           
      
         # ABSORPTION IMAGING
        
        for i in range(5):
            
            # INITIALISATION
            
            # Start with all AOMs on and at MOT settings.
            
            self.dds0.set(80*MHz,amplitude=0.32) # 0.32 #
            self.dds1.set(103*MHz,amplitude=0.42) # 0.42
            self.dds2.set(80*MHz,amplitude=0.32) # 0.32 
            self.dds3.set(226.5*MHz,amplitude=0.31) # 0.31 # Cooling AOM, -1x2 order
            self.dds4.set(234*MHz,amplitude=0.3) # 0.30 #Probe AOM, -1x2 order
            self.dds5.set(110*MHz,amplitude=0.45) # 0.45 #Repump -1x2 Order
           
            self.dds0.sw.on()
            self.dds1.sw.on()
            self.dds2.sw.on()
            self.dds3.sw.on()
            self.dds4.sw.on() 
            self.dds5.sw.on() 
        

            self.coolshutter.on()
            self.eomshutter.on()
            self.probeshutter.off()
            
            self.motcoiltrigger.off()# trigger off == coils on
            self.verticalshim.off() #.off == coil on
            
            delay(100*ms)
            self.dds3.sw.off()
            self.motcoiltrigger.on()    # .on()== coil off 
         
            
            delay(200*ms)
            
                       
            # MOT LOADING
            self.dds3.sw.on()
            self.motcoiltrigger.off()   # Turn on MOT coil 
            
            delay(5*s) # MOT loading time
            
            # #Take an image of the MOT
            # self.cameratrigger.on()
            # delay(1000*us)
            # self.cameratrigger.off() #First image
            
            delay(700*ms)

           ## SUB-DOPPLER COOLING
         #    steps = 1 # Number of steps for the ramp 
         #    Initial_freq = 226.5 # Cooling beam at MOT stage
         #    Final_freq = Initial_freq + 15*i #At the end of Sub-Doppler
         #    freq_step = (Final_freq - Initial_freq)/steps  # MHz - Frequency step for cooling beam ramp
         #    time_step = 7.0 # ms - Each step time 
         #    amp_step =(0.31-0.21)/steps #Initial- final Amp / steps
         # #   print("Total freq changed this run: ", 2*freq_step*steps)
            self.motcoiltrigger.on()
            self.verticalshim.on()   
            delay(10*us)   # off delay for MOT coils - minimum 5.5*us
            
         #    for x in range(steps):                           
         #        self.dds3.set((226.5 + freq_step)*MHz,amplitude=0.31 - amp_step) # 0.31
         #        self.dds5.set((110-freq_step)*MHz,amplitude=0.45)
         #        delay(time_step*ms)
            self.dds3.set(234.5*MHz,amplitude=0.2) # 0.31
            self.dds5.set(103*MHz,amplitude=0.45)
            delay(8*ms)    
            self.dds3.set(238.5*MHz,amplitude=0.2) # 0.31
            self.dds5.set(100*MHz,amplitude=0.45)
            delay(4*ms)   
            # TIME OF FLIGHT
            self.dds3.sw.off()
            delay((459-247)*ns)     # difference in delay times between AOMs 5 and 3 (AOM3 is slower)  
            delay(247*ns)
            
            # self.dds3.set(226.5*MHz,amplitude=0.31) # 0.31
            # delay((50)*ms) # Switch Off time
            
            self.dds3.set(226.5*MHz,amplitude=0.31) # 0.31
            self.dds5.set(110*MHz,amplitude=0.45)
            
            
            delay(38*ms)
            
            
            self.dds3.sw.on()
            
            self.cameratrigger.on()
            delay(1000*us)
            self.cameratrigger.off()
            
            # delay(10*ms)
            
            # self.dds3.sw.off()
            # delay(100*ms)
            
            
            # self.dds3.sw.on()
            # delay(1*ms)
            # self.cameratrigger.on()
            # delay(1000*us)
            # self.cameratrigger.off()
            
            delay(10*ms)
            self.verticalshim.off()
            delay(10*ms)

            
            
            
  