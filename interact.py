from session import turn_level_session
import time
if __name__=='__main__':
    sys=turn_level_session('experiments_21/turn-level-DS-97_34-otl/best_score_model', interacting=True, device1=0)
    while(1):
        user=input("User:")
        if user=='exit':
            print('****Dialog ended****')
            sys.init_session()
            continue
        st=time.time()
        bspn, db, aspn, resp = sys.response(user)
        print('Inference time:{:.3f}s'.format(time.time()-st))
        # delete special token
        print('  bspn:', ' '.join(bspn.split()[1:-1]))
        print('  db:',' '.join(db.split()[1:-1]))
        print('  aspn:', ' '.join(aspn.split()[1:-1]))
        print('Sys:', resp)