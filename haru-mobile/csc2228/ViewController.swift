//
//  ViewController.swift
//  csc2228
//
//  Created by Wenyang Liu on 2017-10-10.
//  Copyright © 2017 csc2228. All rights reserved.
//

import UIKit
import RAMPaperSwitch


class ViewController: UITableViewController {

    @IBOutlet var switch1: RAMPaperSwitch!
    @IBOutlet var switch2: RAMPaperSwitch!
    @IBOutlet var switch3: RAMPaperSwitch!
    @IBOutlet var switch4: RAMPaperSwitch!
    @IBOutlet var switch5: RAMPaperSwitch!
    private var service: Service!
    var switchArray=[RAMPaperSwitch]()

    override func viewDidLoad() {
        super.viewDidLoad()
        service=Service()
        switchArray.append(switch1)
        switchArray.append(switch2)
        switchArray.append(switch3)
        switchArray.append(switch4)
        switchArray.append(switch5)
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    

    @IBAction func `switch`(_ sender: RAMPaperSwitch) {
        
        //For some reasons, the status is regarded `off` when it is on :) Good luck reading this.
        if !sender.isOn {
            disableAll(ramSwitch:sender)
            switch sender {
            case switch1:
                print("===== Mode activated: =====\n"+Constants.SWITCH_STILL)
                service.newTimerTask(action:Constants.SWITCH_STILL)
                break
            case switch2:
                print("===== Mode activated: =====\n"+Constants.SWITCH_WALKING)
                service.newTimerTask(action: Constants.SWITCH_WALKING)
                break
            case switch3:
                print("===== Mode activated: =====\n"+Constants.SWITCH_RUNNING)
                service.newTimerTask(action: Constants.SWITCH_RUNNING)
                break
            case switch4:
                print("===== Mode activated: =====\n"+Constants.SWITCH_GO_UP)
                service.newTimerTask(action: Constants.SWITCH_GO_UP)
                break
            case switch5:
                print("===== Mode activated: =====\n"+Constants.SWITCH_GO_DOWN)
                service.newTimerTask(action: Constants.SWITCH_GO_DOWN)
                break

            default:
                print("Can you tell me how to trigger a non-existing button?")
                break
            }
            
        }
        else{
            service.sendExistingData()
            service.cleanAll()
            enableAll()
        }
        
      
        
        
    }
    
    func disableAll(ramSwitch:RAMPaperSwitch){
        for sw in switchArray {
            if sw == ramSwitch {
                continue
            }
            else{
                sw.isEnabled=false
            }
        }
    }
    
    func enableAll(){
        for sw in switchArray {
          sw.isEnabled=true
        }
    }
    
    


}

