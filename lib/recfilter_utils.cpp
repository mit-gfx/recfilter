#include "recfilter.h"

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::ostream;
using std::runtime_error;

using namespace Halide;

Arguments::Arguments(int argc, char** argv) :
    width  (4096),
    block  (32),
    iterations(1),
    threads(192),
    nocheck(false)
{
    string app_name = argv[0];
    string desc = "\nUsage\n " + app_name + " ";
    desc.append(string(
                "[-width|-w w] [-tile|-block|-b|-t b] [-thread n] [-iter i] [-nocheck] [-help]\n\n"
                "\twidth    width of input image [default = 4096]\n"
                "\ttile     tile width for splitting each dimension image [default = 32]\n"
                "\tthread   maximum threads per tile [default = 192]\n"
                "\tnocheck  do not check against reference solution [default = false]\n"
                "\titer     number of profiling iterations [default = 1]\n"
                "\thelp     show help message\n"
                )
            );

    try {
        for (int i=1; i<argc; i++) {
            string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw runtime_error("Showing help message");
            }

            else if (!option.compare("-nocheck") || !option.compare("--nocheck")) {
                nocheck = true;
            }

            else if (!option.compare("-iter") || !option.compare("--iter")) {
                if ((i+1) < argc)
                    iterations = atoi(argv[++i]);
                else
                    throw runtime_error("-iter requires a int value");
            }

            else if (!option.compare("-thread") || !option.compare("--thread")) {
                if ((i+1) < argc)
                    threads = atoi(argv[++i]);
                else
                    throw runtime_error("-thread requires a int value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc) {
                    width = atoi(argv[++i]);
                }
                else
                    throw runtime_error("-width requires an integer value");
            }

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block") ||
                     !option.compare("-t") || !option.compare("--t") || !option.compare("-tile") || !option.compare("--tile"))
            {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw runtime_error("-block requires an integer value");
            }

            else {
                throw runtime_error("Bad command line option " + option);
            }
        }

        if (width%block)
            throw runtime_error("Width should be a multiple of block size");

    } catch (runtime_error& e) {
        cerr << endl << e.what() << endl << desc << endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------------

ostream &operator<<(ostream &s, CheckResult v) {
    Image<float> ref = v.ref;
    Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Image<double> diff(width, height, channels);

    double re      = 0.0;
    double max_re  = 0.0;
    double mean_re = 0.0;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = ref(x,y,z) - out(x,y,z);
                re       = std::abs(diff(x,y,z)) / ref(x,y,z);
                mean_re += re;
                max_re   = std::max(re, max_re);
            }
        }
    }
    mean_re /= double(width*height*channels);

    s << "Max  relative error = " << 100.0*max_re << " % \n";
    s << "Mean relative error = " << 100.0*mean_re << " % \n\n";

    return s;
}

ostream &operator<<(ostream &s, CheckResultVerbose v) {
    Image<float> ref = v.ref;
    Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Image<double> diff(width, height, channels);

    double re      = 0.0;
    double max_re  = 0.0;
    double mean_re = 0.0;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = (ref(x,y,z) - out(x,y,z));
                re       = std::abs(diff(x,y,z)) / ref(x,y,z);
                mean_re += re;
                max_re   = std::max(re, max_re);
            }
        }
    }
    mean_re /= double(width*height*channels);

    s << "Reference" << "\n" << ref << "\n";
    s << "Halide output" << "\n" << out << "\n";
    s << "Difference " << "\n" << diff << "\n";
    s << "Max  relative error = " << 100.0*max_re << " % \n";
    s << "Mean relative error = " << 100.0*mean_re << " % \n\n";

    return s;
}

ostream &operator<<(ostream &s, RecFilter r) {
    vector<Func> f = r.funcs();
    for (int i=0; i<f.size(); i++) {
        s << f[i] << endl;
    }
    return s;
}

ostream &operator<<(ostream &s, Func f) {
    s << f.function();
    return s;
}

ostream &operator<<(ostream &s, Internal::Function f) {
    if (f.has_pure_definition()) {

        s << "Func " << f.name() << "(\"" << f.name() << "\");\n";

        for (int v=0; v<f.values().size(); v++) {
            vector<string> args = f.args();
            s << f.name() << "(";
            for (int i=0; i<args.size(); i++) {
                s << args[i];
                if (i<args.size()-1)
                    s << ", ";
            }
            if (f.values().size()>1)
                s << ")[" << v << "]";
            else
                s << ")";
            s << " = " << f.values()[v] << ";\n";
        }

        // update definitions
        for (int j=0; j<f.updates().size(); j++) {
            vector<Expr> update_value = f.updates()[j].values;
            for (int v=0; v<update_value.size(); v++) {
                vector<Expr> args = f.updates()[j].args;
                s << f.name() << "(";
                for (int i=0; i<args.size(); i++) {
                    s << args[i];
                    if (i<args.size()-1)
                        s << ", ";
                }
                if (update_value.size()>1)
                    s << ")[" << v << "]";
                else
                    s << ")";
                s << " = " << update_value[v] << ";";
                if (f.updates()[j].domain.defined()) {
                    s << " with  ";
                    for (int k=0; k<f.updates()[j].domain.domain().size(); k++) {
                        string r = f.updates()[j].domain.domain()[k].var;
                        s << r << "("
                            << f.updates()[j].domain.domain()[k].min   << ","
                            << f.updates()[j].domain.domain()[k].extent<< ") ";
                    }
                }
                s << "\n";
            }
        }
    }
    return s;
}

// -----------------------------------------------------------------------------


Timer::Timer(string name) {
    m_Name = name;
    start();
}

Timer::~Timer(void) {
    stop();
}

void Timer::start(void) {
    m_TmStart = milliseconds();
}

Timer::t_time Timer::stop(void) {
    Timer::t_time tm = elapsed();
    Timer::t_time h  = ( tm/(1000*60*60));
    Timer::t_time m  = ((tm/(1000*60)) % 60);
    Timer::t_time s  = ((tm/1000)      % 60);
    Timer::t_time ms = tm % 1000;
    cerr << m_Name.c_str() << ": " << h << "h " << m
        << "m " << s << "s " << ms << "ms" << endl;
    return tm;
}

Timer::t_time Timer::elapsed(void) {
    return (milliseconds() - m_TmStart);
}

Timer::t_time Timer::milliseconds(void) {
    static bool init = false;
#ifdef WIN32
    static Timer::t_time freq;
    if (!init) {
        init = true;
        LARGE_INTEGER lfreq;
        assert(QueryPerformanceFrequency(&lfreq) != 0);
        freq = Timer::t_time(lfreq.QuadPart);
    }
    LARGE_INTEGER tps;
    QueryPerformanceCounter(&tps);
    return (Timer::t_time(tps.QuadPart)*1000/freq);
#else
    struct timeval        now;
    static struct timeval start;
    if (!init) {
        gettimeofday(&start, NULL);
        init = true;
    }
    gettimeofday(&now, NULL);
    uint ms = uint((now.tv_sec-start.tv_sec)*1000+(now.tv_usec-start.tv_usec)/1000);
    return ms;
#endif
}
